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

from CsvDataset import CsvDataset, CsvInferenceDataset
from models.ConvTasNet_models import enhance_ConvTasNet
from models.SpeqGNN import SpeqGNN
from models.SpeqGNN_encoder import SpeqGNN_encoder
from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType
from models.Speq_UNet import Speq_UNet as U_Net
from mymodule import my_func, const, LossFunction, confirmation_GPU
from evaluation import CSV_eval
from models.WaveUnet import Wave_UNet

# CUDA縺ｮ繝｡繝｢繝ｪ邂｡逅・ｨｭ螳・
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDA縺ｮ蜿ｯ逕ｨ諤ｧ繧偵メ繧ｧ繝・け
device = confirmation_GPU.get_device()
print(f"main_Speq 菴ｿ逕ｨ繝・ヰ繧､繧ｹ: {device}")


def padding_tensor(tensor1, tensor2):
	"""
	譛蠕後・谺｡蜈・ｼ井ｾ・ 譎らｳｻ蛻鈴聞・峨′逡ｰ縺ｪ繧・縺､縺ｮ繝・Φ繧ｽ繝ｫ縺ｫ蟇ｾ縺励※縲・
	遏ｭ縺・婿繧呈忰蟆ｾ縺ｫ繧ｼ繝ｭ繝代ョ繧｣繝ｳ繧ｰ縺励※髟ｷ縺輔ｒ縺昴ｍ縺医ｋ縲・

	Args:
		tensor1, tensor2 (torch.Tensor): 莉ｻ諢上・谺｡蜈・焚縺ｮ繝・Φ繧ｽ繝ｫ

	Returns:
		padded_tensor1, padded_tensor2 (torch.Tensor)
	"""
	len1 = tensor1.size(-1)
	len2 = tensor2.size(-1)
	max_len = max(len1, len2)

	pad1 = [0, max_len - len1]  # 譛蠕後・谺｡蜈・□縺代ヱ繝・ぅ繝ｳ繧ｰ
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
	"""GPU縺ｮ險ｭ螳・""
	device = confirmation_GPU.get_device()
	""" 縺昴・莉悶・險ｭ螳・"""
	out_path = Path(out_path)  # path蝙九↓螟画鋤
	out_name, out_dir = out_path.stem, out_path.parent  # 繝輔ぃ繧､繝ｫ蜷阪→繝・ぅ繝ｬ繧ｯ繝医Μ繧貞・髮｢
	# log縺ｮ菫晏ｭ伜・縺ｮ謖・ｮ・"tensorboard --logdir ./logs"縺ｧ遒ｺ隱阪〒縺阪ｋ)
	writer = SummaryWriter(log_dir=f"{const.LOG_DIR}\\{out_name}")

	now = my_func.get_now_time()
	csv_path = os.path.join(const.LOG_DIR, out_name, f"{out_name}_{now}.csv")  # CSV繝輔ぃ繧､繝ｫ縺ｮ繝代せ
	my_func.make_dir(csv_path)
	with open(csv_path, "w") as csv_file:  # 繝輔ぃ繧､繝ｫ繧ｪ繝ｼ繝励Φ
		csv_file.write(f"dataset,out_name,loss_func\n{train_csv},{out_path},{loss_type}")

	""" Early_Stopping縺ｮ險ｭ螳・"""
	best_loss = np.inf  # 謳榊､ｱ髢｢謨ｰ縺ｮ譛蟆丞喧縺檎岼逧・・蝣ｴ蜷茨ｼ悟・繧√・best_loss繧堤┌髯仙､ｧ縺ｫ縺吶ｋ
	earlystopping_count = 0

	""" Load dataset 繝・・繧ｿ繧ｻ繝・ヨ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ """
	train_dataset = CsvDataset(csv_path=train_csv, input_column_header=wave_type, max_length_sec=6)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

	val_dataset = CsvDataset(csv_path=val_csv, input_column_header=wave_type, max_length_sec=6)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

	# print(f"\nmodel:{model}\n")                           # 繝｢繝・Ν縺ｮ繧｢繝ｼ繧ｭ繝・け繝√Ε縺ｮ蜃ｺ蜉・
	""" 譛驕ｩ蛹夜未謨ｰ縺ｮ險ｭ螳・"""
	optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer繧帝∈謚・Adam)

	# torchmetrics繧堤畑縺・◆謳榊､ｱ髢｢謨ｰ縺ｮ蛻晄悄蛹・
	loss_func = LossFunction.get_loss_computer(loss_type, device)

	""" 繝√ぉ繝・け繝昴う繝ｳ繝医・險ｭ螳・"""
	if checkpoint_path != None:
		print("restart_training")
		checkpoint = torch.load(checkpoint_path)  # checkpoint縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
		model.load_state_dict(checkpoint["model_state_dict"])  # 蟄ｦ鄙帝比ｸｭ縺ｮ繝｢繝・Ν縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 繧ｪ繝励ユ繧｣繝槭う繧ｶ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
		# optimizer縺ｮstate繧堤樟蝨ｨ縺ｮdevice縺ｫ遘ｻ縺吶ゅ％繧後ｒ縺励↑縺・→縲∽ｿ晏ｭ伜燕蠕後〒device縺ｮ荳肴紛蜷医′襍ｷ縺薙ｋ蜿ｯ閭ｽ諤ｧ縺後≠繧九・
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device)
		start_epoch = checkpoint["epoch"] + 1
		loss = checkpoint["loss"]
	else:
		start_epoch = 1

	""" 蟄ｦ鄙偵・險ｭ螳壹ｒ蜃ｺ蜉・"""
	print("====================")
	print("device: ", device)
	print("out_path: ", out_path)
	print("dataset: ", train_csv)
	print("loss_func: ", loss_type)
	print("accumulation_steps: ", accumulation_steps)
	print("====================")

	my_func.make_dir(out_dir)
	model.train()  # 蟄ｦ鄙偵Δ繝ｼ繝峨↓險ｭ螳・

	start_time = time.time()  # 譎る俣繧呈ｸｬ螳・
	epoch = 0
	for epoch in range(start_epoch, train_count + 1):  # 蟄ｦ鄙貞屓謨ｰ
		print("Train Epoch:", epoch)  # 蟄ｦ鄙貞屓謨ｰ縺ｮ陦ｨ遉ｺ
		model_loss_sum = 0  # 邱乗錐螟ｱ縺ｮ蛻晄悄蛹・
		optimizer.zero_grad()
		for i, (mix_data, target_data) in tenumerate(train_loader):
			mix_data, target_data = mix_data.to(device), target_data.to(device)  # 繝・・繧ｿ繧竪PU縺ｫ遘ｻ蜍・

			""" 繝・・繧ｿ縺ｮ謨ｴ蠖｢ """
			mix_data = mix_data.to(torch.float32)  # target_data縺ｮ繧ｿ繧､繝励ｒ螟画鋤 int16竊断loat32
			target_data = target_data.to(torch.float32)  # target_data縺ｮ繧ｿ繧､繝励ｒ螟画鋤 int16竊断loat32

			""" 繝｢繝・Ν縺ｫ騾壹☆(莠域ｸｬ蛟､縺ｮ險育ｮ・ """
			estimate_data = model(mix_data)  # 繝｢繝・Ν縺ｫ騾壹☆

			""" 繝・・繧ｿ縺ｮ謨ｴ蠖｢ """
			estimate_data, target_data = padding_tensor(estimate_data, target_data)
			# target_data = target_data.squeeze(dim=1)  # (B, 1, length)
			# estimate_data = estimate_data.unsqueeze(dim=1)  # (B, 1, length)

			""" 謳榊､ｱ縺ｮ險育ｮ・"""
			# print("estimate_data shape:", estimate_data.shape)
			# print("target_data shape:", target_data.shape)
			model_loss = loss_func(estimate_data, target_data)
			model_loss = model_loss / accumulation_steps

			""" 蠕悟・逅・"""
			model_loss.backward()  # 隱､蟾ｮ騾・ｼ晄成
			model_loss_sum += model_loss.item() * accumulation_steps

			if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
				optimizer.step()  # 蜍ｾ驟阪・譖ｴ譁ｰ
				optimizer.zero_grad()

			del (
				mix_data,
				target_data,
				model_loss,
			)  # 菴ｿ逕ｨ縺励※縺・↑縺・､画焚縺ｮ蜑企勁 estimate_data,
			torch.cuda.empty_cache()  # 繝｡繝｢繝ｪ縺ｮ隗｣謾ｾ 1iteration縺斐→縺ｫ隗｣謾ｾ

		""" 繝√ぉ繝・け繝昴う繝ｳ繝医・菴懈・ """
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
		print(f"[{epoch}]model_loss_sum:{model_loss_sum}")  # 謳榊､ｱ縺ｮ蜃ｺ蜉・

		torch.cuda.empty_cache()  # 繝｡繝｢繝ｪ縺ｮ隗｣謾ｾ 1iteration縺斐→縺ｫ隗｣謾ｾ
		with open(csv_path, "a") as out_file:  # 繝輔ぃ繧､繝ｫ繧ｪ繝ｼ繝励Φ
			out_file.write(f"{model_loss_sum}\n")  # 譖ｸ縺崎ｾｼ縺ｿ

		""" Early_Stopping 縺ｮ蛻､譁ｭ """
		model.eval()
		val_loss = 0.0

		# 蜍ｾ驟崎ｨ育ｮ励ｒ辟｡蜉ｹ蛹悶＠縺ｦ繝｡繝｢繝ｪ蜉ｹ邇・ｒ荳翫￡繧・
		with torch.no_grad():
			progress_bar_val = tqdm(val_loader, desc="Validation")
			for mix_data, target_data in progress_bar_val:
				mix_data = mix_data.to(device)
				target_data = target_data.to(device)

				estimate_data = model(mix_data)

				estimate_data, target_data = padding_tensor(estimate_data, target_data)
				# estimate_data = estimate_data.unsqueeze(dim=1)  # (B, 1, length)
				model_loss = loss_func(estimate_data, target_data)
				val_loss += model_loss.item()
				progress_bar_val.set_postfix({"loss": model_loss.item()})
			avg_val_loss = val_loss / len(val_loader)
		if avg_val_loss < best_loss:
			print(f"Validation loss improved ({best_loss:.6f} --> {avg_val_loss:.6f}). Saving model...")
			best_loss = avg_val_loss
			# 譛濶ｯ繝｢繝・Ν繧剃ｿ晏ｭ・
			torch.save(model.state_dict(), f"{out_dir}/BEST_{out_name}.pth")
			earlystopping_count = 0  # 繧ｫ繧ｦ繝ｳ繧ｿ繝ｼ繧偵Μ繧ｻ繝・ヨ
		else:
			earlystopping_count += 1
			print(f"Validation loss did not improve. Patience: {earlystopping_count}/{earlystopping_threshold}")

		if earlystopping_count >= earlystopping_threshold:
			print("Early stopping triggered. Training finished.")
			break

	torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")  # 蜃ｺ蜉帙ヵ繧｡繧､繝ｫ縺ｮ菫晏ｭ・

	""" 蟄ｦ鄙偵Δ繝・Ν(pth繝輔ぃ繧､繝ｫ)縺ｮ蜃ｺ蜉・"""
	print("model save")
	torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")  # 蜃ｺ蜉帙ヵ繧｡繧､繝ｫ縺ｮ菫晏ｭ・

	writer.close()

	""" 蟄ｦ鄙呈凾髢薙・險育ｮ・"""
	time_end = time.time()  # 迴ｾ蝨ｨ譎る俣縺ｮ蜿門ｾ・
	time_sec = time_end - start_time  # 邨碁℃譎る俣縺ｮ險育ｮ・sec)
	time_h = float(time_sec) / 3600.0  # sec->hour
	print(f"time・嘴str(time_h)}h")  # 蜃ｺ蜉・


def test(model: nn.Module, test_csv: str, wave_type: str, out_dir: str, model_path: str, prm: int = const.SR):
	# 繝・ぅ繝ｬ繧ｯ繝医Μ繧剃ｽ懈・
	my_func.make_dir(out_dir)
	model_path = Path(model_path)  # path蝙九↓螟画鋤
	model_dir, model_name = (
		model_path.parent,
		model_path.stem,
	)  # 繝輔ぃ繧､繝ｫ蜷阪→繝・ぅ繝ｬ繧ｯ繝医Μ繧貞・髮｢

	model.load_state_dict(torch.load(os.path.join(model_dir, f"BEST_{model_name}.pth"), map_location=device))
	model.eval()

	dataset = CsvInferenceDataset(csv_path=test_csv, input_column_header=wave_type)
	dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

	for mix_data, mix_name in tqdm(dataset_loader):
		mix_data = mix_data.to(device)  # 繝・・繧ｿ繧竪PU縺ｫ遘ｻ蜍・
		mix_data = mix_data.to(torch.float32)  # 繝・・繧ｿ縺ｮ蝙九ｒ螟画鋤 int16竊断loat32

		separate = model(mix_data)  # 繝｢繝・Ν縺ｮ驕ｩ逕ｨ
		# print(f"Initial separate shape: {separate.shape}") # 繝・ヰ繝・げ逕ｨ

		separate = separate.cpu()
		separate = separate.detach().numpy()
		# print(f"separate: {separate.shape}")
		# print(f"mix_name: {mix_name}")
		# print(f"mix_name: {type(mix_name)}")

		# separate 縺ｮ蠖｢迥ｶ繧・(length,) 縺ｫ謨ｴ蠖｢縺吶ｋ
		# 繝｢繝・Ν縺ｮ蜃ｺ蜉帙′ (1, 1, length) 縺ｨ莉ｮ螳・
		data_to_write = separate.squeeze()

		# 豁｣隕丞喧
		mix_max = torch.max(mix_data)  # mix_wave縺ｮ譛螟ｧ蛟､繧貞叙蠕・
		data_to_write = data_to_write / np.max(data_to_write) * mix_max.cpu().detach().numpy()

		# 蛻・屬縺励◆ speech繧貞・蜉帙ヵ繧｡繧､繝ｫ縺ｨ縺励※菫晏ｭ倥☆繧九・
		# 繝輔ぃ繧､繝ｫ蜷阪→繝輔か繝ｫ繝蜷阪ｒ邨仙粋縺励※繝代せ譁・ｭ怜・繧剃ｽ懈・
		out_path = os.path.join(out_dir, (mix_name[0] + ".wav"))
		# print('saving... ', fname)
		# 豺ｷ蜷医ョ繝ｼ繧ｿ繧剃ｿ晏ｭ・
		# my_func.save_wav(out_path, separate[0], prm)
		sf.write(out_path, data_to_write, prm)
		torch.cuda.empty_cache()  # 繝｡繝｢繝ｪ縺ｮ隗｣謾ｾ 1髻ｳ螢ｰ縺斐→縺ｫ隗｣謾ｾ


if __name__ == "__main__":
	"""繝｢繝・Ν縺ｮ險ｭ螳・""
	num_mic = 1  # 繝槭う繧ｯ縺ｮ謨ｰ
	num_node = 32  # 繝弱・繝峨・謨ｰ
	model_list = [
		"UNet"
	]  # 繝｢繝・Ν縺ｮ遞ｮ鬘・ "UGCN", "UGCN2", "UGAT", "UGAT2", "ConvTasNet", "UNet"
	wave_types = [
		"noise_reverb",
		"reverb_only",
		"noise_only",
	]  # 蜈･蜉帑ｿ｡蜿ｷ縺ｮ遞ｮ鬘・(noise_only, reverbe_only, noise_reverb)

	node_selection = NodeSelectionType.ALL  # 繝弱・繝蛾∈謚槭・譁ｹ豕・(ALL, TEMPORAL)
	edge_selection = EdgeSelectionType.GRID  # 繧ｨ繝・ず驕ｸ謚槭・譁ｹ豕・(RAMDOM, KNN, GRID)

	graph_config = GraphConfig(
		num_edges=num_node,
		node_selection=node_selection,
		edge_selection=edge_selection,
		bidirectional=True,
		temporal_window=4000,  # 譎る俣遯薙・繧ｵ繧､繧ｺ
	)
	stft_params = {
		"n_fft": 512,
		"hop_length": 256,
		"win_length": 512
	}

	for model_type in model_list:
		if model_type == "GCN":
			model = SpeqGNN(n_channels=num_mic, n_classes=num_mic, gnn_type="GCN", graph_config=graph_config, **stft_params).to(device)
		elif model_type == "GAT":
			model = SpeqGNN(n_channels=num_mic, n_classes=num_mic, gnn_type="GAT", graph_config=graph_config, **stft_params).to(device)
		elif model_type == "GCNEncoder":
			model = SpeqGNN_encoder(n_channels=num_mic, gnn_type="GCN", num_node=num_node, graph_config=graph_config).to(device)
		elif model_type == "GATEncoder":
			model = SpeqGNN_encoder(n_channels=num_mic, gnn_type="GAT", num_node=num_node, graph_config=graph_config).to(device)
		elif model_type == "ConvTasNet":
			model = enhance_ConvTasNet().to(device)
		elif model_type == "UNet":
			model = Wave_UNet(num_inputs=num_mic, num_outputs=1).to(device)
		else:
			raise ValueError(f"Unknown model type: {model_type}")

		dir_name = "Random_Dataset_VCTK_DEMAND_1ch"  # 繝・・繧ｿ繧ｻ繝・ヨ縺ｮ繝・ぅ繝ｬ繧ｯ繝医Μ蜷・
		loss_type = "SISDR"  # 謳榊､ｱ髢｢謨ｰ縺ｮ遞ｮ鬘・("SISDR", "wave_MSE", "stft_MSE")
		model_type = f"Wave{model_type}"
		for wave_type in wave_types:
			# out_name = f"new_{model_type}_{wave_type}_{num_node}node_{node_selection.value}_{edge_selection.value}"  # 蜃ｺ蜉帛錐
			out_name = f"{model_type}_{wave_type}"  # 蜃ｺ蜉帛錐
			# C:\Users\kataoka-lab\Desktop\sound_data\sample_data\speech\DEMAND\clean\train
			train(model=model,
			      train_csv=f"{const.MIX_DATA_DIR}/{dir_name}/train.csv",
			      val_csv=f"{const.MIX_DATA_DIR}/{dir_name}/val.csv",
			      wave_type=wave_type,
			      out_path=f"{const.PTH_DIR}/{dir_name}/{model_type}/{out_name}.pth",
			      loss_type=loss_type,
			      batchsize=16, checkpoint_path=None, train_count=500, earlystopping_threshold=10, accumulation_steps=1)

			test(model=model,
			     test_csv=f"{const.MIX_DATA_DIR}/{dir_name}/test.csv",
			     wave_type=wave_type,
			     out_dir=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}",
			     model_path=f"{const.PTH_DIR}/{dir_name}/{model_type}/{out_name}.pth")

			CSV_eval.main(input_csv_path=f"{const.MIX_DATA_DIR}/{dir_name}/test.csv",
			              target_column="clean",
			              estimation_column=wave_type,
			              estimation_dir=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}",
			              out_path=f"{const.EVALUATION_DIR}/{dir_name}/{model_type}/{out_name}_CSV.csv")
