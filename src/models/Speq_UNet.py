import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from mymodule import confirmation_GPU

# CUDAの可用性をチェック
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = confirmation_GPU.get_device()
print(f"Speq_UNet.py 使用デバイス: {device}")


class DoubleConv(nn.Module):
	""" (畳み込み => バッチ正規化 => ReLU) を2回繰り返すブロック """

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	""" ダウンサンプリングブロック (マックスプーリング + DoubleConv) """

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class Up(nn.Module):
	""" アップサンプリングブロック (転置畳み込み + DoubleConv) """

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)


class Speq_UNet(nn.Module):
	def __init__(self, n_channels=1, n_classes=1, n_fft=512, hop_length=256, win_length=None):
		super(Speq_UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes

		# ISTFT parameters
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.win_length = win_length if win_length is not None else n_fft
		self.window = torch.hann_window(self.win_length)

		# 3-layer U-Net
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.up1 = Up(256, 128)
		self.up2 = Up(128, 64)
		self.outc = OutConv(64, n_classes)

	def forward(self, x_magnitude, complex_spec_input, original_length=None):
		"""
		Args:
			x_magnitude (torch.Tensor): Input magnitude spectrogram [Batch, Channels, Freq, Time]
			complex_spec_input (torch.Tensor): Input complex spectrogram [Batch, Freq, Time]
			original_length (int, optional): The original length of the time-domain signal for ISTFT.
		"""
		# The input is already a spectrogram, so we extract phase from the complex spectrogram
		phase = torch.angle(complex_spec_input)  # (B, F, T)

		# U-Net
		x1 = self.inc(x_magnitude)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		d2 = self.up1(x3, x2)
		d1 = self.up2(d2, x1)

		# Mask prediction
		mask_pred_raw = self.outc(d1)
		mask_pred = torch.sigmoid(mask_pred_raw)

		# Resize mask to match input magnitude size
		if mask_pred.shape[2:] != x_magnitude.shape[2:]:
			mask_pred = F.interpolate(mask_pred, size=x_magnitude.shape[2:], mode='bilinear', align_corners=False)

		# Apply mask
		predicted_magnitude = x_magnitude * mask_pred

		# ISTFT
		predicted_magnitude_for_istft = predicted_magnitude.squeeze(1)  # (B, F, T)
		reconstructed_complex_spec = torch.polar(predicted_magnitude_for_istft, phase)

		output_waveform = torch.istft(reconstructed_complex_spec,
									  n_fft=self.n_fft,
									  hop_length=self.hop_length,
									  win_length=self.win_length,
									  window=self.window.to(reconstructed_complex_spec.device),
									  return_complex=False,
									  length=original_length)
		return output_waveform


def print_model_summary(model, batch_size, freq_bins, time_frames, original_length):
	# For summary, input needs to be on the correct device.
	x_mag = torch.randn(batch_size, model.n_channels, freq_bins, time_frames).to(device)
	x_complex = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.cfloat).to(device)

	print(f"\n--- Speq_UNet Model Summary ---")
	summary(model, input_data=(x_mag, x_complex, original_length), device=device)


def main():
	print("Speq_UNet.py main execution")
	batch = 8
	length = 16000 * 8  # 4 seconds of audio

	# STFT parameters
	n_fft = 512
	hop_length = n_fft // 2
	win_length = n_fft
	window = torch.hann_window(win_length, device=device)

	# --- Create dummy spectrogram input ---
	# Create dummy time-domain signal to get realistic spectrograms
	x_time = torch.randn(batch, length, device=device)
	x_complex_spec = torch.stft(x_time, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
								return_complex=True)
	x_magnitude_spec = torch.abs(x_complex_spec).unsqueeze(1)  # Add channel dimension

	# Update dimensions based on actual STFT output
	_b, _f, _t = x_complex_spec.shape
	freq_bins = _f
	time_frames = _t

	model = Speq_UNet(n_channels=1, n_classes=1, n_fft=n_fft, hop_length=hop_length).to(device)

	print_model_summary(model, batch, freq_bins, time_frames, length)

	# Test forward pass
	print(f"\nInput magnitude shape: {x_magnitude_spec.shape}")
	print(f"Input complex spec shape: {x_complex_spec.shape}")
	output = model(x_magnitude_spec, x_complex_spec, original_length=length)
	print(f"Output shape: {output.shape}")

	if torch.cuda.is_available():
		print(f"\nGPU Memory Usage:")
		print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
		print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == '__main__':
	main()
