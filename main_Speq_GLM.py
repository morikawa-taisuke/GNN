import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib import tenumerate

from All_evaluation import main as evaluation
from CsvDataset import CsvDataset, CsvInferenceDataset
# from models.ConvTasNet_models import enhance_ConvTasNet # Speq_GLMでは不要
from models.Speq_GLM import SpeqGNN  # ★ 変更点: SpeqGNN -> Speq_GLM
# from models.SpeqGNN_encoder import SpeqGNN_encoder # Speq_GLMでは不要
# from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType # Speq_GLMでは不要
# from models.Speq_UNet import Speq_UNet as U_Net # Speq_GLMでは不要
from mymodule import my_func, const, LossFunction, confirmation_GPU
import CSV_eval

# CUDAのメモリ管理設定
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAの可用性をチェック
device = confirmation_GPU.get_device()
print(f"main_Speq_GLM 使用デバイス: {device}")


def padding_tensor(tensor1, tensor2):
	"""
	最後の次元（例: 時系列長）が異なる2つのテンソルに対して、
	短い方を末尾にゼロパディングして長さをそろえる。

	Args:
		tensor1, tensor2 (torch.Tensor): 任意の次元数のテンソル

	Returns:
		padded_tensor1, padded_tensor2 (torch.Tensor)
	"""
	len1 = tensor1.size(-1)
	len2 = tensor2.size(-1)
	max_len = max(len1, len2)

	pad1 = [0, max_len - len1]  # 最後の次元だけパディング
	pad2 = [0, max_len - len2]

	padded_tensor1 = F.pad(tensor1, pad1)
	padded_tensor2 = F.pad(tensor2, pad2)

	return padded_tensor1, padded_tensor2


def train(model: nn.Module,
          train_csv: str,
          val_csv: str,
          wave_type: str,
          out_path: str = "./RESULT/pth/result.pth",
          loss_type: str = "stft_MSE",
          batchsize: int = const.BATCHSIZE,
          checkpoint_path: str = None,
          train_count: int = const.EPOCH,
          earlystopping_threshold: int = 5,
          accumulation_steps: int = 4,
          graph_reg_lambda: float = 0.1):  # ★ 変更点: グラフ正則化損失の重みを追加
	"""GPUの設定"""
	device = confirmation_GPU.get_device()
	""" その他の設定 """
	out_path = Path(out_path)  # path型に変換
	out_name, out_dir = out_path.stem, out_path.parent  # ファイル名とディレクトリを分離
	# logの保存先の指定("tensorboard --logdir ./logs"で確認できる)
	writer = SummaryWriter(log_dir=f"{const.LOG_DIR}\\{out_name}")

	now = my_func.get_now_time()
	csv_path = os.path.join(const.LOG_DIR, out_name, f"{out_name}_{now}.csv")  # CSVファイルのパス
	my_func.make_dir(csv_path)
	with open(csv_path, "w") as csv_file:  # ファイルオープン
		# ★ 変更点: graph_reg_lambda をCSVヘッダに記録
		csv_file.write(f"dataset,out_name,loss_func,graph_reg_lambda\n{train_csv},{out_path},{loss_type},{graph_reg_lambda}\n")
		csv_file.write(
			f"epoch,total_loss,model_loss,graph_loss,val_total_loss,val_model_loss,val_graph_loss\n")  # ★ 変更点: 損失の内訳を記録

	""" Early_Stoppingの設定 """
	best_loss = np.inf  # 損失関数の最小化が目的の場合，初めのbest_lossを無限大にする
	earlystopping_count = 0

	""" Load dataset データセットの読み込み """
	train_dataset = CsvDataset(csv_path=train_csv, input_column_header=wave_type, max_length_sec=6)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True,
	                          collate_fn=CsvDataset.collate_fn)

	val_dataset = CsvDataset(csv_path=val_csv, input_column_header=wave_type, max_length_sec=6)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True, pin_memory=True,
	                        collate_fn=CsvDataset.collate_fn)

	# print(f"\nmodel:{model}\n")                           # モデルのアーキテクチャの出力
	""" 最適化関数の設定 """
	optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizerを選択(Adam)

	# torchmetricsを用いた損失関数の初期化
	loss_func = LossFunction.get_loss_computer(loss_type, device)

	""" チェックポイントの設定 """
	if checkpoint_path != None:
		print("restart_training")
		checkpoint = torch.load(checkpoint_path)  # checkpointの読み込み
		model.load_state_dict(checkpoint["model_state_dict"])  # 学習途中のモデルの読み込み
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # オプティマイザの読み込み
		# optimizerのstateを現在のdeviceに移す。これをしないと、保存前後でdeviceの不整合が起こる可能性がある。
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device)
		start_epoch = checkpoint["epoch"] + 1
		loss = checkpoint["loss"]
	else:
		start_epoch = 1

	""" 学習の設定を出力 """
	print("====================")
	print("device: ", device)
	print("out_path: ", out_path)
	print("dataset: ", train_csv)
	print("loss_func: ", loss_type)
	print("graph_reg_lambda: ", graph_reg_lambda)  # ★ 変更点: ラムダを表示
	print("accumulation_steps: ", accumulation_steps)
	print("====================")

	my_func.make_dir(out_dir)
	model.train()  # 学習モードに設定

	start_time = time.time()  # 時間を測定
	epoch = 0
	for epoch in range(start_epoch, train_count + 1):  # 学習回数
		print("Train Epoch:", epoch)  # 学習回数の表示

		# ★ 変更点: 損失を分けて集計
		total_loss_sum = 0.0
		model_loss_sum = 0.0
		graph_loss_sum = 0.0

		optimizer.zero_grad()
		for i, (mix_data, target_data) in tenumerate(train_loader):
			mix_data, target_data = mix_data.to(device), target_data.to(device)  # データをGPUに移動

			""" データの整形 """
			mix_data = mix_data.to(torch.float32)  # target_dataのタイプを変換 int16→float32
			target_data = target_data.to(torch.float32)  # target_dataのタイプを変換 int16→float32

			""" モデルに通す(予測値の計算) """
			# --- STFT ---
			original_length = mix_data.shape[-1]
			# torchaudio.stftは (batch, time) または (time) を期待するため、チャンネル次元を削除
			mix_data_squeezed = mix_data.squeeze(1)

			# 複素スペクトログラムを計算
			mix_complex = torch.stft(
				mix_data_squeezed,
				n_fft=model.n_fft,
				hop_length=model.hop_length,
				win_length=model.win_length,
				window=model.window.to(device),
				return_complex=True
			)
			mix_magnitude = torch.abs(mix_complex).unsqueeze(1)  # (B, 1, F, T)
			estimate_data = model(mix_magnitude, mix_complex, original_length)  # モデルに通す

			""" データの整形 """
			estimate_data, target_data = padding_tensor(estimate_data, target_data)
			target_data = target_data.squeeze(dim=1)  # (B, 1, length)
			# estimate_data = estimate_data.unsqueeze(dim=1)  # (B, 1, length)

			""" 損失の計算 """
			# ★ 変更点: グラフ損失を考慮
			model_loss_only = loss_func(estimate_data, target_data)
			graph_loss = model.latest_graph_reg_loss  # モデルからグラフ損失を取得
			total_loss = model_loss_only + graph_reg_lambda * graph_loss

			total_loss_acc = total_loss / accumulation_steps  # 勾配蓄積のために割る

			""" 後処理 """
			total_loss_acc.backward()  # 誤差逆伝搬 (合計損失で)

			# ★ 変更点: 損失を分けて集計 (蓄積前の値)
			total_loss_sum += total_loss.item()
			model_loss_sum += model_loss_only.item()
			graph_loss_sum += graph_loss.item()

			if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
				optimizer.step()  # 勾配の更新
				optimizer.zero_grad()

			del (
				mix_data,
				target_data,
				model_loss_only,
				graph_loss,
				total_loss,
				total_loss_acc
			)  # 使用していない変数の削除
			torch.cuda.empty_cache()  # メモリの解放 1iterationごとに解放

		""" チェックポイントの作成 """
		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"loss": total_loss_sum / len(train_loader),  # 平均合計損失
			},
			f"{out_dir}/{out_name}_ckp.pth",
		)

		# ★ 変更点: 損失のロギングを詳細化
		avg_total_loss = total_loss_sum / len(train_loader)
		avg_model_loss = model_loss_sum / len(train_loader)
		avg_graph_loss = graph_loss_sum / len(train_loader)

		writer.add_scalar(f"Loss/Train_Total", avg_total_loss, epoch)
		writer.add_scalar(f"Loss/Train_Model", avg_model_loss, epoch)
		writer.add_scalar(f"Loss/Train_Graph", avg_graph_loss, epoch)

		print(f"[{epoch}] Train Total Loss: {avg_total_loss:.4f} (Model: {avg_model_loss:.4f}, Graph: {avg_graph_loss:.4f})")

		torch.cuda.empty_cache()  # メモリの解放

		""" Early_Stopping の判断 """
		model.eval()
		# ★ 変更点: 検証損失も分けて集計
		val_total_loss_sum = 0.0
		val_model_loss_sum = 0.0
		val_graph_loss_sum = 0.0

		# 勾配計算を無効化してメモリ効率を上げる
		with torch.no_grad():
			progress_bar_val = tqdm(val_loader, desc="Validation")
			for mix_data, target_data in progress_bar_val:
				mix_data = mix_data.to(device)
				target_data = target_data.to(device)

				# --- STFT ---
				original_length = mix_data.shape[-1]
				mix_data_squeezed = mix_data.squeeze(1)

				mix_complex = torch.stft(
					mix_data_squeezed,
					n_fft=model.n_fft,
					hop_length=model.hop_length,
					win_length=model.win_length,
					window=model.window.to(device),
					return_complex=True
				)
				mix_magnitude = torch.abs(mix_complex).unsqueeze(1)

				estimate_data = model(mix_magnitude, mix_complex, original_length)

				estimate_data, target_data = padding_tensor(estimate_data, target_data)

				# ★ 変更点: 損失計算 (GLM対応)
				model_loss_only = loss_func(estimate_data, target_data)
				graph_loss = model.latest_graph_reg_loss
				total_loss = model_loss_only + graph_reg_lambda * graph_loss

				val_total_loss_sum += total_loss.item()
				val_model_loss_sum += model_loss_only.item()
				val_graph_loss_sum += graph_loss.item()

				progress_bar_val.set_postfix({"loss": total_loss.item()})

			# ★ 変更点: 平均検証損失の計算とロギング
			avg_val_total_loss = val_total_loss_sum / len(val_loader)
			avg_val_model_loss = val_model_loss_sum / len(val_loader)
			avg_val_graph_loss = val_graph_loss_sum / len(val_loader)

			writer.add_scalar(f"Loss/Val_Total", avg_val_total_loss, epoch)
			writer.add_scalar(f"Loss/Val_Model", avg_val_model_loss, epoch)
			writer.add_scalar(f"Loss/Val_Graph", avg_val_graph_loss, epoch)

		# CSVにエポックごとの損失を書き込み
		with open(csv_path, "a") as out_file:  # ファイルオープン
			out_file.write(
				f"{epoch},{avg_total_loss},{avg_model_loss},{avg_graph_loss},{avg_val_total_loss},{avg_val_model_loss},{avg_val_graph_loss}\n")

		# ★ 変更点: Early Stoppingは合計損失 (avg_val_total_loss) で判断
		if avg_val_total_loss < best_loss:
			print(f"Validation loss improved ({best_loss:.6f} --> {avg_val_total_loss:.6f}). Saving model...")
			best_loss = avg_val_total_loss
			# 最良モデルを保存
			torch.save(model.state_dict(), f"{out_dir}/BEST_{out_name}.pth")
			earlystopping_count = 0  # カウンターをリセット
		else:
			earlystopping_count += 1
			print(f"Validation loss did not improve. Patience: {earlystopping_count}/{earlystopping_threshold}")

		if earlystopping_count >= earlystopping_threshold:
			print("Early stopping triggered. Training finished.")
			break

		model.train()  # 次のエポックのためにモデルを訓練モードに戻す

	torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")  # 出力ファイルの保存

	""" 学習モデル(pthファイル)の出力 """
	print("model save")
	torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")  # 出力ファイルの保存

	writer.close()

	""" 学習時間の計算 """
	time_end = time.time()  # 現在時間の取得
	time_sec = time_end - start_time  # 経過時間の計算(sec)
	time_h = float(time_sec) / 3600.0  # sec->hour
	print(f"time：{str(time_h)}h")  # 出力


def test(model: nn.Module, test_csv: str, wave_type: str, out_dir: str, model_path: str, prm: int = const.SR):
	# (test関数は main_Speq.py と同一。変更不要)

	# ディレクトリを作成
	my_func.make_dir(out_dir)
	model_path = Path(model_path)  # path型に変換
	model_dir, model_name = (
		model_path.parent,
		model_path.stem,
	)  # ファイル名とディレクトリを分離

	# ★ 変更点: BESTモデルを読み込む
	best_model_path = os.path.join(model_dir, f"BEST_{model_name}.pth")
	if not os.path.exists(best_model_path):
		print(f"Warning: BEST model not found at {best_model_path}. Falling back to latest checkpoint.")
		best_model_path = os.path.join(model_dir, f"{model_name}.pth")  # .pth が元々の名前

	model.load_state_dict(torch.load(best_model_path, map_location=device))
	model.eval()

	dataset = CsvInferenceDataset(csv_path=test_csv, input_column_header=wave_type)
	dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)  # shuffle=False

	for mix_data, mix_name in tqdm(dataset_loader):
		mix_data = mix_data.to(device)  # データをGPUに移動
		mix_data = mix_data.to(torch.float32)  # データの型を変換 int16→float32

		# --- STFT ---
		original_length = mix_data.shape[-1]
		mix_data_squeezed = mix_data.squeeze(1)

		mix_complex = torch.stft(
			mix_data_squeezed,
			n_fft=model.n_fft,
			hop_length=model.hop_length,
			win_length=model.win_length,
			window=model.window.to(device),
			return_complex=True
		)
		mix_magnitude = torch.abs(mix_complex).unsqueeze(1)

		with torch.no_grad():  # 推論時は勾配計算不要
			separate = model(mix_magnitude, mix_complex, original_length)  # モデルの適用

		separate = separate.cpu()
		separate = separate.detach().numpy()

		data_to_write = separate.squeeze()

		# 正規化 (オプション)
		# mix_max = torch.max(mix_data)
		# data_to_write = data_to_write / np.max(data_to_write) * mix_max.cpu().detach().numpy()

		# 分離した speechを出力ファイルとして保存する。
		out_path = os.path.join(out_dir, (mix_name[0] + ".wav"))
		sf.write(out_path, data_to_write, prm)
		torch.cuda.empty_cache()  # メモリの解放 1音声ごとに解放


if __name__ == "__main__":
	"""モデルの設定"""
	num_mic = 1  # マイクの数

	# ★ 変更点: Speq_GLM がサポートするモデルタイプ
	model_list = [
		"GAT"
	]  # モデルの種類  "GCN", "GAT"

	wave_types = [
		"noise_only",
		"reverb_only",
		"noise_reverb",
	]  # 入力信号の種類 (noise_only, reverbe_only, noise_reverbe)

	# ★ 変更点: GLM用の設定
	glm_k = 16  # Graph Learning Module の k (近傍)
	graph_reg_lambda = 0.1  # グラフ正則化損失の重み
	hidden_dim = 32  # GNNの隠れ層の次元
	gat_heads = 4  # GATのヘッド数
	gat_dropout = 0.6  # GATのドロップアウト率

	stft_params = {
		"n_fft": 512,
		"hop_length": 256,
		"win_length": 512
	}

	for model_type in model_list:
		# ★ 変更点: Speq_GLM の SpeqGNN を初期化
		if model_type == "GCN":
			model = SpeqGNN(n_channels=num_mic, n_classes=num_mic,
			                gnn_type="GCN",
			                glm_k=glm_k,
			                hidden_dim=hidden_dim,
			                **stft_params).to(device)
		elif model_type == "GAT":
			model = SpeqGNN(n_channels=num_mic, n_classes=num_mic,
			                gnn_type="GAT",
			                glm_k=glm_k,
			                hidden_dim=hidden_dim,
			                gat_heads=gat_heads,
			                gat_dropout=gat_dropout,
			                **stft_params).to(device)
		else:
			raise ValueError(f"Unknown model type: {model_type}")

		dir_name = "DEMAND_hoth_10dB_500msec"  # データセットのディレクトリ名
		loss_type = "SISDR"  # 損失関数の種類 ("SISDR", "wave_MSE", "stft_MSE")

		# ★ 変更点: モデル名にGLMとkを追加
		model_name_base = f"Speq_{model_type}_GLM_k{glm_k}"

		for wave_type in wave_types:
			out_name = f"{model_name_base}_{wave_type}"  # 出力名

			print(f"\n--- Starting Training for {out_name} ---")

			train(model=model,
			      train_csv=f"{const.MIX_DATA_DIR}/{dir_name}/train.csv",
			      val_csv=f"{const.MIX_DATA_DIR}/{dir_name}/val.csv",
			      wave_type=wave_type,
			      out_path=f"{const.PTH_DIR}/{dir_name}/{model_name_base}/{out_name}.pth",
			      loss_type=loss_type,
			      batchsize=1,
			      checkpoint_path=None,
			      train_count=500,
			      earlystopping_threshold=10,
			      accumulation_steps=16,
			      graph_reg_lambda=graph_reg_lambda)  # ★ 変更点: ラムダを渡す

			print(f"\n--- Starting Testing for {out_name} ---")

			test(model=model,
			     test_csv=f"{const.MIX_DATA_DIR}/{dir_name}/test.csv",
			     wave_type=wave_type,
			     out_dir=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_name_base}/{out_name}",
			     model_path=f"{const.PTH_DIR}/{dir_name}/{model_name_base}/{out_name}.pth")

			print(f"\n--- Starting Evaluation for {out_name} ---")

			CSV_eval.main(input_csv_path=f"{const.MIX_DATA_DIR}/{dir_name}/test.csv",
			              target_column="clean",
			              estimation_column=wave_type,  # この列は実際には使われないが、CsvInferenceDatasetとの互換性のため
			              estimation_dir=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_name_base}/{out_name}",
			              out_path=f"{const.EVALUATION_DIR}/{dir_name}/{model_name_base}/{out_name}_CSV.csv")