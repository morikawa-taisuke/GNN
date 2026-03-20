import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR
# Import torchmetrics for loss functions
from torchmetrics.regression import MeanSquaredError as MSE


def get_loss_computer(loss_func_name: str, device: torch.device):
	"""
	損失関数名に基づいて、損失を計算する関数（callable）を返す。
	ループ内の分岐をなくし、コードをクリーンにする。

	Args:
		loss_func_name (str): 損失関数の名前 ("SISDR", "wave_MSE", "stft_MSE")
		device (torch.device): 計算に使用するデバイス

	Returns:
		callable: (preds, target) -> loss を計算する関数
	"""
	if loss_func_name == "SISDR":
		metric = SISDR().to(device)
		# SI-SDRは値が大きいほど良いため、損失として最小化するには-1を掛ける
		return lambda preds, target: -metric(preds, target)

	elif loss_func_name == "wave_MSE":
		metric = MSE().to(device)
		# MSEはそのまま損失として使える
		return metric

	elif loss_func_name == "stft_MSE":
		metric = MSE().to(device)

		def stft_mse_computer(preds, target):
			# STFTは (batch, signal_length) を期待するため、チャンネル次元を削除
			# モデルの出力が (batch, 1, length) であることを想定
			stft_preds = torch.stft(preds.squeeze(1), n_fft=1024, return_complex=False)
			stft_target = torch.stft(target.squeeze(1), n_fft=1024, return_complex=False)
			return metric(stft_preds, stft_target)

		return stft_mse_computer
	else:
		raise ValueError(f"Unknown loss function: {loss_func_name}")

if __name__ == "__main__":
	# Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Create sample data
	batch_size = 2
	signal_length = 16000
	preds = torch.randn(batch_size, 1, signal_length).to(device)
	target = torch.randn(batch_size, 1, signal_length).to(device)


	def test_sisdr():
		loss_computer = get_loss_computer("SISDR", device)
		loss = loss_computer(preds, target)
		print(f"SISDR loss: {loss.item()}")


	def test_wave_mse():
		loss_computer = get_loss_computer("wave_MSE", device)
		loss = loss_computer(preds, target)
		print(f"Wave MSE loss: {loss.item()}")


	def test_stft_mse():
		loss_computer = get_loss_computer("stft_MSE", device)
		loss = loss_computer(preds, target)
		print(f"STFT MSE loss: {loss.item()}")


	print("Testing loss functions...")
	test_sisdr()
	test_wave_mse()
	test_stft_mse()

	try:
		get_loss_computer("invalid_loss", device)
	except ValueError as e:
		print(f"Successfully caught error: {e}")
