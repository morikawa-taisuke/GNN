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

# 既存の評価スクリプトをインポート
# from All_evaluation import main as evaluation
# 修正済みデータローダーをインポート
from CsvDataset import ReverbEncoderDataset, CsvInferenceDataset
# 補助損失計算用のヘルパーをインポート
from mymodule import my_func, const, LossFunction, confirmation_GPU
from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType


from models.Graph_Encoder import ReverbGNNEncoder

# CUDAの可用性をチェック
device = confirmation_GPU.get_device()
print(f"Using device: {device}")


def padding_tensor(tensor1, tensor2):
	"""
	最後の次元（例: 時系列長）が異なる2つのテンソルに対して、
	短い方を末尾にゼロパディングして長さをそろえる。
	"""
	len1 = tensor1.size(-1)
	len2 = tensor2.size(-1)
	max_len = max(len1, len2)

	pad1 = [0, max_len - len1]
	pad2 = [0, max_len - len2]

	padded_tensor1 = F.pad(tensor1, pad1)
	padded_tensor2 = F.pad(tensor2, pad2)

	return padded_tensor1, padded_tensor2


def train(model: nn.Module,
          train_csv: str,
          val_csv: str,
          wave_type: str,
          reverb_loss_weight: float = 0.1,  # ★追加: 補助損失の重み (α)
          out_path: str = "./RESULT/pth/result.pth",
          main_loss_type: str = "SISDR",  # ★変更: main_loss_type に名称変更
          batchsize: int = const.BATCHSIZE,
          checkpoint_path: str = None,
          train_count: int = const.EPOCH,
          earlystopping_threshold: int = 5):
	"""学習関数 (マルチタスク学習対応)
	Args:
		model (nn.Module): 学習させるモデル (ReverbGNNEncoder)
		reverb_feature_columns (list): 教師残響特徴量として使用するCSVの列名リスト
		reverb_loss_weight (float): 補助損失の重み α
		main_loss_type (str): 主損失関数の種類 ("stft_MSE", "L1", "MSE", "SISDR")
        // ... (その他省略)
	"""

	"""GPUの設定"""
	device = confirmation_GPU.get_device()

	""" その他の設定 """
	out_path = Path(out_path)
	out_name, out_dir = out_path.stem, out_path.parent
	writer = SummaryWriter(log_dir=f"{const.LOG_DIR}\\{out_name}")

	now = my_func.get_now_time()
	csv_path = os.path.join(const.LOG_DIR, out_name, f"{out_name}_{now}.csv")
	my_func.make_dir(csv_path)
	with open(csv_path, "w") as csv_file:
		csv_file.write(
			f"dataset,out_name,main_loss_func,reverb_loss_weight\n{train_csv},{out_path},{main_loss_type},{reverb_loss_weight}\n")
		csv_file.write("epoch,total_loss,main_loss,reverb_loss\n")  # ログヘッダーを修正

	""" Early_Stoppingの設定 """
	best_loss = np.inf
	earlystopping_count = 0

	""" Load dataset データセットの読み込み (★reverb_feature_columnsを追加) """
	train_dataset = ReverbEncoderDataset(csv_path=train_csv, input_column_header=wave_type)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True,
	                          pin_memory=True, collate_fn=ReverbEncoderDataset.collate_fn)

	val_dataset = ReverbEncoderDataset(csv_path=val_csv, input_column_header=wave_type)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True,
	                        pin_memory=True, collate_fn=ReverbEncoderDataset.collate_fn)

	""" 最適化関数の設定 """
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	# 主損失 (強調音声の品質)
	main_loss_func = LossFunction.get_loss_computer(main_loss_type, device)
	# 補助損失 (残響特徴量の精度) - 標準のMSEを使用
	reverb_loss_func = nn.MSELoss().to(device)

	""" チェックポイントの設定 (既存ロジックは省略) """
	# ... (必要に応じて実装)

	""" 学習の設定を出力 (既存ロジックは省略) """
	# ...

	my_func.make_dir(out_dir)
	model.train()

	start_time = time.time()
	for epoch in range(1, train_count + 1):
		print(f"Train Epoch: {epoch}")
		total_loss_sum = 0
		main_loss_sum = 0
		reverb_loss_sum = 0

		# ★変更: dataloaderから3つの要素をアンパック
		for _, (mix_data, target_data, reverb_true) in tenumerate(train_loader):

			# ★変更: reverb_true をGPUに移動
			mix_data, target_data, reverb_true = mix_data.to(device), target_data.to(device), reverb_true.to(device)

			optimizer.zero_grad()

			mix_data = mix_data.to(torch.float32)
			target_data = target_data.to(torch.float32)

			# ★変更: モデルから2つの出力を取得
			estimate_data_w, reverb_pred = model(mix_data)

			# データの整形
			estimate_data_w, target_data = padding_tensor(estimate_data_w, target_data)

			# モデルの出力 (waveform) にチャンネル次元がない場合に追加 (通常 [B, 1, L] が必要)
			if estimate_data_w.ndim == 2:
				estimate_data_w = estimate_data_w.unsqueeze(1)

			# --- 損失の計算 ---
			# 1. 主損失 (強調音声)
			L_main = main_loss_func(estimate_data_w, target_data)

			# 2. 補助損失 (残響特徴量)
			L_reverb = reverb_loss_func(reverb_pred, reverb_true)

			# 3. 総損失
			model_loss = L_main + reverb_loss_weight * L_reverb

			# --- 損失の集計 ---
			total_loss_sum += model_loss.item()
			main_loss_sum += L_main.item()
			reverb_loss_sum += L_reverb.item() * reverb_loss_weight  # 重みを掛けた後の値を集計

			""" 後処理 """
			model_loss.backward()
			optimizer.step()

			del mix_data, target_data, reverb_true, estimate_data_w, reverb_pred, model_loss
			torch.cuda.empty_cache()

		# --- エポック集計とログ ---
		avg_total_loss = total_loss_sum / len(train_loader)
		avg_main_loss = main_loss_sum / len(train_loader)
		avg_reverb_loss = reverb_loss_sum / len(train_loader)

		# チェックポイントの作成 (既存ロジックは省略)
		torch.save({
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"loss": avg_total_loss,  # 総損失を保存
		}, f"{out_dir}/{out_name}_ckp.pth")

		writer.add_scalar(f'total_loss', avg_total_loss, epoch)
		writer.add_scalar(f'L_main', avg_main_loss, epoch)
		writer.add_scalar(f'L_reverb', avg_reverb_loss, epoch)

		print(f"[{epoch}] Total Loss: {avg_total_loss:.6f}, Main Loss: {avg_main_loss:.6f}, Reverb Loss: {avg_reverb_loss:.6f}")

		torch.cuda.empty_cache()
		with open(csv_path, "a") as out_file:
			out_file.write(f"{epoch},{avg_total_loss},{avg_main_loss},{avg_reverb_loss}\n")

		""" Early_Stopping の判断 (既存ロジックを総損失で実行) """
		model.eval()
		val_total_loss = 0.0

		with torch.no_grad():
			progress_bar_val = tqdm(val_loader, desc="Validation")
			# ★変更: dataloaderから3つの要素をアンパック
			for mix_data, target_data, reverb_true in progress_bar_val:
				mix_data, target_data, reverb_true = mix_data.to(device), target_data.to(device), reverb_true.to(device)

				estimate_data_w, reverb_pred = model(mix_data)
				estimate_data_w, target_data = padding_tensor(estimate_data_w, target_data)
				if estimate_data_w.ndim == 2:
					estimate_data_w = estimate_data_w.unsqueeze(1)

				L_main = main_loss_func(estimate_data_w, target_data)
				L_reverb = reverb_loss_func(reverb_pred, reverb_true)
				model_loss = L_main + reverb_loss_weight * L_reverb

				val_total_loss += model_loss.item()
				progress_bar_val.set_postfix({"loss": model_loss.item()})

			avg_val_loss = val_total_loss / len(val_loader)

		if avg_val_loss < best_loss:
			print(f"Validation loss improved ({best_loss:.6f} --> {avg_val_loss:.6f}). Saving model...")
			best_loss = avg_val_loss
			torch.save(model.state_dict(), f"{out_dir}/BEST_{out_name}.pth")
			earlystopping_count = 0
		else:
			earlystopping_count += 1
			print(f"Validation loss did not improve. Patience: {earlystopping_count}/{earlystopping_threshold}")

		if earlystopping_count >= earlystopping_threshold:
			print("Early stopping triggered. Training finished.")
			break

	torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")

	""" 学習モデル(pthファイル)の出力 (既存ロジックは省略) """
	# ...
	writer.close()

	""" 学習時間の計算 (既存ロジックは省略) """
	time_end = time.time()  # 現在時間の取得
	time_sec = time_end - start_time  # 経過時間の計算(sec)
	time_h = float(time_sec) / 3600.0  # sec->hour
	print(f"time：{str(time_h)}h")  # 出力


def test(model: nn.Module, test_csv: str, wave_type: str, out_dir: str, model_path: str, prm: int = const.SR):
	"""
	推論関数 (強調音声のみを出力)
	"""
	# ディレクトリを作成
	my_func.make_dir(out_dir)
	model_path = Path(model_path)
	model_dir, model_name = model_path.parent, model_path.stem

	# モデルのロード
	model.load_state_dict(torch.load(os.path.join(model_dir, f"BEST_{model_name}.pth"), map_location=device))
	model.eval()

	# CsvInferenceDataset は波形とファイル名のみを返すため、reverb_feature_columns は不要
	dataset = CsvInferenceDataset(csv_path=test_csv, input_column_header=wave_type)
	dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)  # 推論はshuffle=Falseが一般的

	for mix_data, mix_name in tqdm(dataset_loader):
		mix_data = mix_data.to(device).to(torch.float32)

		# ★変更: モデルから強調音声 (estimate_data_w) のみを取得 (reverb_pred は破棄)
		with torch.no_grad():
			estimate_data_w, _ = model(mix_data)

		separate = estimate_data_w.cpu().squeeze().detach().numpy()

		# 正規化 (既存ロジック)
		mix_max = torch.max(mix_data).cpu().detach().numpy()
		if np.max(np.abs(separate)) > 1e-8:  # ゼロ除算防止
			data_to_write = separate / np.max(np.abs(separate)) * mix_max
		else:
			data_to_write = separate  # 全て0の場合はそのまま

		# 保存
		out_path = os.path.join(out_dir, (mix_name[0] + ".wav"))
		sf.write(out_path, data_to_write, prm)
		torch.cuda.empty_cache()


if __name__ == "__main__":
	# --- 学習実行設定 ---
	num_mic = 1
	num_node = 16
	model_type = "ReverbGNNEncoder"  # 新しいモデル名
	gnn_type = "GCN"
	wave_type = "noise_reverb_path"  # CSVの入力カラム名

	# ★補助タスク設定 (重要)
	# CsvDataset.pyで定義したカラム名と一致させる
	reverb_cols = ["cepstrum_coeffs", "rt60", "c50", "d50"]
	# 総特徴次元 = ケプストラム(16) + RT60(1) + C50(1) + D50(1) = 19
	reverb_dim = 19
	reverb_loss_weight = 0.1  # 補助損失の重み α

	# 🚨 パスの設定 🚨
	# const.MIX_DATA_DIR が適切に設定されていることを前提とします。
	# ここでは、CSVと出力のベースディレクトリを仮定します。
	dir_name = "reverb_encoder"
	train_csv_path = f"{const.MIX_DATA_DIR}/{dir_name}/mix_wav/train.csv"
	val_csv_path = f"{const.MIX_DATA_DIR}/{dir_name}/mix_wav/val.csv"
	test_csv_path = f"{const.MIX_DATA_DIR}/{dir_name}/mix_wav/test.csv"

	# --- モデル初期化 ---
	# GraphConfigは必要に応じて調整してください
	graph_config = GraphConfig(
		num_edges=num_node,
		node_selection=NodeSelectionType.ALL,
		edge_selection=EdgeSelectionType.KNN,
		bidirectional=True,
	)

	model = ReverbGNNEncoder(
		n_channels=num_mic,
		n_classes=1,
		num_node=num_node,
		gnn_type=gnn_type,
		graph_config=graph_config,
		reverb_feature_dim=reverb_dim  # 特徴量次元を渡す
	).to(device)

	# --- 学習の実行 ---
	out_name = f"{model_type}_{gnn_type}_alpha{reverb_loss_weight}_node{num_node}"

	train(
		model=model,
		train_csv=train_csv_path,
		val_csv=val_csv_path,
		wave_type=wave_type,
	    reverb_loss_weight=reverb_loss_weight, # ★追加
		out_path=f"{const.PTH_DIR}/{dir_name}/{model_type}/{out_name}.pth",
		main_loss_type="SISDR",
		batchsize=4, checkpoint_path=None, train_count=500, earlystopping_threshold=10
	)

	# --- 推論の実行 ---
	test_out_dir = f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}"
	test(
		model=model,
		test_csv=test_csv_path,
		wave_type=wave_type,
		out_dir=test_out_dir,
		model_path=f"{const.PTH_DIR}/{dir_name}/{model_type}/{out_name}.pth"
	)

	# --- 評価の実行 ---
	# evaluation(
	# 	target_dir=f"path/to/your/{dir_name}/test/clean", # クリーン音声のディレクトリを指すように修正
	# 	estimation_dir=test_out_dir,
	# 	out_path=f"{const.EVALUATION_DIR}/{dir_name}/{model_type}/{out_name}.csv",
	# )

	print("--- スクリプト生成完了 ---")
	print("🚨 実行前に、モデルファイルのインポート、CSV/ディレクトリパス、そして `const.py` のパス設定を確認してください。")