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

	try:
		if input_data is not None:
			summary(model, input_data=input_data, device=device, **kwargs)
		elif input_size is not None:
			summary(model, input_size=input_size, device=device, **kwargs)
		else:
			print("Warning: No input_size or input_data provided. Summary might be incomplete.")
			summary(model, device=device, **kwargs)
	except Exception as e:
		print(f"Error generating summary: {e}")

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

	# --- サンプルモデルでの実行例 ---
	class SampleModel(nn.Module):
		def __init__(self):
			super().__init__()
			self.conv = nn.Conv2d(1, 16, 3, padding=1)
			self.fc = nn.Linear(16 * 10 * 10, 10)

		def forward(self, x):
			x = F.relu(self.conv(x))
			x = x.view(x.size(0), -1)
			return self.fc(x)

	print("\n--- Sample Model Check ---")
	model = SampleModel().to(device)
	# 入力サイズ: (Batch, Channels, H, W)
	input_size = (1, 1, 10, 10)

	# print_model_summary を呼ぶだけで、サマリーとメモリ使用量が両方表示されます
	print_model_summary(model, input_size=input_size, device=device)

	# --- (参考) 実際のモデルをチェックする場合の例 ---
	# from models.SpeqGNN import SpeqGNN
	# model = SpeqGNN(...).to(device)
	# print_model_summary(model, input_size=(1, 1, 257, 100))


if __name__ == '__main__':
	from models.SpeqGNN import SpeqGNN

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	input_size = (1, 1, 257, 100)   # (Batch, Channels, H, W)
	model = SpeqGNN().to(device)
	main()
