import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def print_model_summary(model, input_size=None, input_data=None, device=None, **kwargs):
	"""
	モデルのサマリーとメモリ使用量を表示する汎用関数。

	Args:
		model (nn.Module): 対象のPyTorchモデル
		input_size (tuple, optional): 入力サイズ (batch_size, channels, ...)
		input_data (tensor or tuple, optional): 実際の入力データ。input_sizeより優先されます。
		device (str or torch.device, optional): デバイス。指定がない場合はモデルのパラメータから推測します。
		**kwargs: torchinfo.summary に渡すその他の引数 (col_names, depthなど)
	"""
	if device is None:
		try:
			device = next(model.parameters()).device
		except StopIteration:
			device = torch.device('cpu')

	print(f"\n{'=' * 20} Model Summary: {model.__class__.__name__} {'=' * 20}")

	# メモリチェック (Before)
	check_memory_usage(device, label="Start")

	summary(model, input_size=input_size, input_data=input_data, device=device, **kwargs)

	# メモリチェック (After)
	check_memory_usage(device, label="End")
	print(f"{'=' * 60}\n")


def check_memory_usage(device, label=""):
	"""GPUメモリ使用量を表示する関数"""
	if isinstance(device, str):
		device = torch.device(device)

	if device.type == 'cuda':
		allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
		reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
		print(f"[{label}] GPU Memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")
	else:
		# CPUの場合は表示しない
		pass


def main(model, input_size=None, input_data=None, device=None, **kwargs):
	"""
	model_checker.py の動作確認用メイン関数。
	実際のモデルのチェックを行いたい場合は、ここでモデルをインスタンス化して
	print_model_summary を呼び出してください。
	"""
	print("Running model_checker main...")
	print(f"Device: {device}")

	# print_model_summary を呼ぶだけで、サマリーとメモリ使用量が両方表示されます
	print_model_summary(model, input_size=input_size, input_data=input_data, device=device)


if __name__ == '__main__':
	from models.Speq_UNet import Speq_UNet
	from models.SpeqGNN import SpeqGNN
	from models.WaveUGNN import Wave_UGNN
	from models.WaveUnet import Wave_UNet
	from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# モデルリスト
	model_type_list = [
		"UNet",
		# "GCN",
		# "GAT",
	]

	# 入力のタイプ
	input_type_list = [
		# "Wave", # 時間領域
		"Spectrogram"   # 周波数領域
	]

	# その他パラメータ
	Batch_size = 2
	num_mic = 1  # マイク数
	wave_time = 15  # sec
	sampling_rate = 16000
	time_length = sampling_rate * wave_time
	num_node = 16  # ノード数
	node_selection = NodeSelectionType.ALL  # ノード選択の方法 (ALL, TEMPORAL)
	edge_selection = EdgeSelectionType.RANDOM  # エッジ選択の方法 (RANDOM, KNN, GRID)

	graph_config = GraphConfig(
		num_edges=num_node,
		node_selection=node_selection,
		edge_selection=edge_selection,
		bidirectional=True,
		temporal_window=4000,  # 時間窓のサイズ
	)
	stft_params = {
		"n_fft": 512,
		"hop_length": 256,
		"win_length": 512
	}
	for model_type in model_type_list:
		for input_type in input_type_list:
			# model = None
			input_size = None
			input_data = None
			if input_type == "Wave":
				if model_type == "UNet":
					model = Wave_UNet(num_inputs=1, num_outputs=1).to(device)
				else:
					model = Wave_UGNN(gnn_type=model_type, graph_config=graph_config).to(device)
				input_size = (Batch_size, num_mic, time_length)  # (Batch, Channels, Length)
			elif input_type == "Spectrogram":
				if model_type == "UNet":
					model = Speq_UNet().to(device)
				else:
					model = SpeqGNN(gnn_type=model_type, graph_config=graph_config, **stft_params).to(device)
				
				# STFTデータの生成
				dummy_wave = torch.randn(Batch_size, time_length, device=device)
				window = torch.hann_window(stft_params["win_length"], device=device)
				
				complex_spec = torch.stft(
					dummy_wave,
					n_fft=stft_params["n_fft"],
					hop_length=stft_params["hop_length"],
					win_length=stft_params["win_length"],
					window=window,
					return_complex=True
				) # (Batch, Freq, Time)
				
				magnitude_spec = torch.abs(complex_spec).unsqueeze(1) # (Batch, 1, Freq, Time)
				
				# SpeqGNN/Speq_UNetのforward引数に合わせてinput_dataを作成
				# forward(self, x_magnitude, complex_spec_input, original_length=None)
				input_data = (magnitude_spec, complex_spec, time_length)
				input_size = None # input_dataを使用するためNoneにする

			main(model, input_size=input_size, input_data=input_data, device=device)
