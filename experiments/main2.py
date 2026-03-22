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

# from All_evaluation import main as evaluation
from CsvDataset import CsvDataset, CsvInferenceDataset
from models.ConvTasNet_models import enhance_ConvTasNet
from models.GNN import UGNN
from models.GNN_encoder import GNNEncoder
from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType
from models.wave_unet import U_Net
from mymodule import my_func, const, LossFunction, confirmation_GPU
from evaluation import CSV_eval

# CUDAのメモリ管琁E��宁E
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAの可用性をチェチE��
device = confirmation_GPU.get_device()
print(f"Using device: {device}")


def padding_tensor(tensor1, tensor2):
	"""
	最後�E次允E��侁E 時系列長�E�が異なめEつのチE��ソルに対して、E
	短ぁE��を末尾にゼロパディングして長さをそろえる、E

	Args:
		tensor1, tensor2 (torch.Tensor): 任意�E次允E��のチE��ソル

	Returns:
		padded_tensor1, padded_tensor2 (torch.Tensor)
	"""
	len1 = tensor1.size(-1)
	len2 = tensor2.size(-1)
	max_len = max(len1, len2)

	pad1 = [0, max_len - len1]  # 最後�E次允E��けパチE��ング
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
		  accumulation_steps: int = 4):
	"""学習関数
	Args:
		model (nn.Module): 学習させるモチE��
		train_csv (str): 学習用CSVファイルのパス
		val_csv (str): 検証用CSVファイルのパス
		wave_type (str): 入力信号の種顁E(noise_only, reverbe_only, noise_reverb)
		out_path (str): 学習後�EモチE��の保存�Eパス
		loss_type (str): 損失関数の種顁E("stft_MSE", "L1", "MSE", "SISDR")
		batchsize (int): バッチサイズ
		checkpoint_path (str): チェチE��ポイント�Eパス�E�Eoneの場合�E新規学習！E
		train_count (int): 学習エポック数
		earlystopping_threshold (int): Early Stoppingの閾値
		accumulation_steps (int): 勾配を蓁E��するスチE��プ数
	"""
	"""GPUの設宁E""
	device = confirmation_GPU.get_device()
	""" そ�E他�E設宁E"""
	out_path = Path(out_path)  # path型に変換
	out_name, out_dir = out_path.stem, out_path.parent  # ファイル名とチE��レクトリを�E離
	# logの保存�Eの持E��E"tensorboard --logdir ./logs"で確認できる)
	writer = SummaryWriter(log_dir=f"{const.LOG_DIR}\\{out_name}")

	now = my_func.get_now_time()
	csv_path = os.path.join(const.LOG_DIR, out_name, f"{out_name}_{now}.csv")  # CSVファイルのパス
	my_func.make_dir(csv_path)
	with open(csv_path, "w") as csv_file:  # ファイルオープン
		csv_file.write(f"dataset,out_name,loss_func\n{train_csv},{out_path},{loss_type}")

	""" Early_Stoppingの設宁E"""
	best_loss = np.inf  # 損失関数の最小化が目皁E�E場合，�Eめ�Ebest_lossを無限大にする
	earlystopping_count = 0

	""" Load dataset チE�EタセチE��の読み込み """
	train_dataset = CsvDataset(csv_path=train_csv, input_column_header=wave_type, max_length_sec=6)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

	val_dataset = CsvDataset(csv_path=val_csv, input_column_header=wave_type)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

	# print(f"\nmodel:{model}\n")                           # モチE��のアーキチE��チャの出劁E
	""" 最適化関数の設宁E"""
	optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizerを選抁EAdam)

	# torchmetricsを用ぁE��損失関数の初期匁E
	loss_func = LossFunction.get_loss_computer(loss_type, device)

	""" チェチE��ポイント�E設宁E"""
	if checkpoint_path != None:
		print("restart_training")
		checkpoint = torch.load(checkpoint_path)  # checkpointの読み込み
		model.load_state_dict(checkpoint["model_state_dict"])  # 学習途中のモチE��の読み込み
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # オプティマイザの読み込み
		# optimizerのstateを現在のdeviceに移す。これをしなぁE��、保存前後でdeviceの不整合が起こる可能性がある、E
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device)
		start_epoch = checkpoint["epoch"] + 1
		loss = checkpoint["loss"]
	else:
		start_epoch = 1

	""" 学習�E設定を出劁E"""
	print("====================")
	print("device: ", device)
	print("out_path: ", out_path)
	print("dataset: ", train_csv)
	print("loss_func: ", loss_type)
	print("accumulation_steps: ", accumulation_steps)
	print("====================")

	my_func.make_dir(out_dir)
	model.train()  # 学習モードに設宁E

	start_time = time.time()  # 時間を測宁E
	epoch = 0
	for epoch in range(start_epoch, train_count + 1):  # 学習回数
		print("Train Epoch:", epoch)  # 学習回数の表示
		model_loss_sum = 0  # 総損失の初期匁E
		optimizer.zero_grad()  # 勾配をエポックの開始時にリセチE��

		for i, (mix_data, target_data) in tenumerate(train_loader):
			mix_data, target_data = mix_data.to(device), target_data.to(device)  # チE�EタをGPUに移勁E

			""" チE�Eタの整形 """
			mix_data = mix_data.to(torch.float32)  # target_dataのタイプを変換 int16→float32
			target_data = target_data.to(torch.float32)  # target_dataのタイプを変換 int16→float32

			""" モチE��に通す(予測値の計箁E """
			estimate_data = model(mix_data)  # モチE��に通す

			""" チE�Eタの整形 """
			estimate_data, target_data = padding_tensor(estimate_data, target_data)

			""" 損失の計箁E"""
			model_loss = loss_func(estimate_data, target_data)

			# 勾配蓄積�Eために損失をスケール
			model_loss = model_loss / accumulation_steps

			""" 誤差送E��播 """
			model_loss.backward()

			# ログ記録用にスケールを戻した損失を加箁E
			model_loss_sum += model_loss.item() * accumulation_steps

			""" 勾配�E更新 """
			if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
				optimizer.step()
				optimizer.zero_grad()

			del (
				mix_data,
				target_data,
				estimate_data,
				model_loss,
			)  # 使用してぁE��ぁE��数の削除
			torch.cuda.empty_cache()  # メモリの解放 1iterationごとに解放

		""" チェチE��ポイント�E作�E """
		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"loss": model_loss_sum,
			},
			f"{out_dir}/{out_name}_ckp.pth",
		)

		writer.add_scalar(str(out_name[0]), model_loss_sum, epoch)
		print(f"[{epoch}]model_loss_sum:{model_loss_sum}")  # 損失の出劁E

		torch.cuda.empty_cache()  # メモリの解放 1iterationごとに解放
		with open(csv_path, "a") as out_file:  # ファイルオープン
			out_file.write(f"{model_loss_sum}\n")  # 書き込み

		""" Early_Stopping の判断 """
		model.eval()
		val_loss = 0.0

		# 勾配計算を無効化してメモリ効玁E��上げめE
		with torch.no_grad():
			progress_bar_val = tqdm(val_loader, desc="Validation")
			for mix_data, target_data in progress_bar_val:
				mix_data = mix_data.to(device)
				target_data = target_data.to(device)

				estimate_data = model(mix_data)
				estimate_data, target_data = padding_tensor(estimate_data, target_data)
				model_loss = loss_func(estimate_data, target_data)
				val_loss += model_loss.item()
				progress_bar_val.set_postfix({"loss": model_loss.item()})
			avg_val_loss = val_loss / len(val_loader)

		if avg_val_loss < best_loss:
			print(f"Validation loss improved ({best_loss:.6f} --> {avg_val_loss:.6f}). Saving model...")
			best_loss = avg_val_loss
			# 最良モチE��を保孁E
			torch.save(model.state_dict(), f"{out_dir}/BEST_{out_name}.pth")
			earlystopping_count = 0  # カウンターをリセチE��
		else:
			earlystopping_count += 1
			print(f"Validation loss did not improve. Patience: {earlystopping_count}/{earlystopping_threshold}")

		if earlystopping_count >= earlystopping_threshold:
			print("Early stopping triggered. Training finished.")
			break

	torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")  # 出力ファイルの保孁E

	""" 学習モチE��(pthファイル)の出劁E"""
	print("model save")
	torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")  # 出力ファイルの保孁E

	writer.close()

	""" 学習時間�E計箁E"""
	time_end = time.time()  # 現在時間の取征E
	time_sec = time_end - start_time  # 経過時間の計箁Esec)
	time_h = float(time_sec) / 3600.0  # sec->hour
	print(f"time�E�{str(time_h)}h")  # 出劁E


def test(model: nn.Module, test_csv: str, wave_type: str, out_dir: str, model_path: str, prm: int = const.SR, out_name: str = "output"):
	# チE��レクトリを作�E
	my_func.make_dir(out_dir)
	model_path = Path(model_path)  # path型に変換
	model_dir, model_name = (
		model_path.parent,
		model_path.stem,
	)  # ファイル名とチE��レクトリを�E離

	model.load_state_dict(torch.load(os.path.join(model_dir, f"BEST_{model_name}.pth"), map_location=device))
	model.eval()

	dataset = CsvInferenceDataset(csv_path=test_csv, input_column_header=wave_type)
	dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

	for mix_data, mix_name in tqdm(dataset_loader):
		mix_data = mix_data.to(device)  # チE�EタをGPUに移勁E
		mix_data = mix_data.to(torch.float32)  # チE�Eタの型を変換 int16→float32

		separate = model(mix_data, export_name=mix_name[0], out_dir=out_name)  # モチE��の適用
		continue
		# print(f"Initial separate shape: {separate.shape}") # チE��チE��用

		separate = separate.cpu()
		separate = separate.detach().numpy()
		# print(f"separate: {separate.shape}")
		# print(f"mix_name: {mix_name}")
		# print(f"mix_name: {type(mix_name)}")

		# separate の形状めE(length,) に整形する
		# モチE��の出力が (1, 1, length) と仮宁E
		data_to_write = separate.squeeze()

		# 正規化
		mix_max = torch.max(mix_data)  # mix_waveの最大値を取征E
		data_to_write = data_to_write / np.max(data_to_write) * mix_max.cpu().detach().numpy()

		# 刁E��した speechを�E力ファイルとして保存する、E
		# ファイル名とフォルダ名を結合してパス斁E���Eを作�E
		out_path = os.path.join(out_dir, (mix_name[0] + ".wav"))
		# print('saving... ', fname)
		# 混合データを保孁E
		# my_func.save_wav(out_path, separate[0], prm)
		sf.write(out_path, data_to_write, prm)
		torch.cuda.empty_cache()  # メモリの解放 1音声ごとに解放


if __name__ == "__main__":
	"""モチE��の設宁E""
	num_mic = 1  # マイクの数
	num_node = 32  # ノ�Eド�E数
	model_list = [
        "UGCN", "UGAT",
	]  # モチE��の種顁E "UGCN", "UGAT", "ConvTasNet", "UNet"
	wave_types = [
		"clean",
		# "reverb_only",
		# "noise_reverb",
	]  # 入力信号の種顁E(noise_only, reverb_only, noise_reverb)    # UGAT_all_random_reverb_only
	node_selection_list = [
		NodeSelectionType.TEMPORAL,
		NodeSelectionType.ALL
	]  # ノ�Eド選択�E方況E(ALL, TEMPORAL)
	edge_selection_list = [
		EdgeSelectionType.RANDOM,
		EdgeSelectionType.KNN
	]  # エチE��選択�E方況E(RANDOM, KNN)

	for node_selection in node_selection_list:
		for edge_selection in edge_selection_list:
			graph_config = GraphConfig(
				num_edges=num_node,
				node_selection=node_selection,
				edge_selection=edge_selection,
				bidirectional=True,
				temporal_window=4000,  # 時間窓�Eサイズ
			)

			for model_type in model_list:
				if model_type == "UGCN":
					model = UGNN(n_channels=num_mic, num_node=num_node, gnn_type="GCN", graph_config=graph_config).to(device)
				elif model_type == "UGAT":
					model = UGNN(n_channels=num_mic, num_node=num_node, gnn_type="GAT", graph_config=graph_config).to(device)
				elif model_type == "GCNEncoder":
					model = GNNEncoder(n_channels=num_mic, gnn_type="GCN", num_node=num_node, graph_config=graph_config).to(device)
				elif model_type == "GATEncoder":
					model = GNNEncoder(n_channels=num_mic, gnn_type="GAT", num_node=num_node, graph_config=graph_config).to(device)
				elif model_type == "ConvTasNet":
					model = enhance_ConvTasNet().to(device)
				elif model_type == "UNet":
					model = U_Net().to(device)
				else:
					raise ValueError(f"Unknown model type: {model_type}")


				dir_name = "Random_Dataset_VCTK_DEMAND_1ch"
				for wave_type in wave_types:
					# out_name = f"{model_type}_{wave_type}"	# 出力名
					out_name = f"{model_type}_{wave_type}_{num_node}node_{node_selection.value}_{edge_selection.value}"  # 出力名
					out_dir = f"{model_type}_{node_selection.value}_{edge_selection.value}_{wave_type}"  # 出力名
					# C:\Users\kataoka-lab\Desktop\sound_data\sample_data\speech\DEMAND\clean\train
					# train(model=model,
					# 	  train_csv=f"{const.MIX_DATA_DIR}/{dir_name}/train.csv",
					# 	  val_csv=f"{const.MIX_DATA_DIR}/{dir_name}/val.csv",
					# 	  wave_type=wave_type,
					# 	  out_path=f"{const.CHECKPOINT_DIR}/{dir_name}/{model_type}/{out_name}.pth",
					# 	  loss_type="SISDR",
					# 	  batchsize=8, checkpoint_path=None, train_count=500, earlystopping_threshold=10, accumulation_steps=2)


					test(model=model,
						 test_csv=f"{const.MIX_DATA_DIR}/{dir_name}/test.csv",
						 wave_type=wave_type,
						 out_dir=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}",
						 model_path=f"{const.CHECKPOINT_DIR}/{dir_name}/{model_type}/{out_name}.pth",
						 out_name=out_dir)

					# evaluation(
					# 	target_dir=f"{const.MIX_DATA_DIR}/{dir_name}/test/clean",
					# 	estimation_dir=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}",
					# 	out_path=f"{const.EVALUATION_DIR}/{dir_name}/{model_type}/{out_name}.csv",
					# )

					# CSV_eval.main(input_csv_path=f"{const.MIX_DATA_DIR}/{dir_name}/test.csv",
					# 			  target_column="clean",
					# 			  estimation_column=wave_type,
					# 			  estimation_dir=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}",
					# 			  out_path=f"{const.EVALUATION_DIR}/{dir_name}/{model_type}/{out_name}.csv")
