import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchinfo import summary

# デバイスの確認（既存コードの mymodule を利用することを想定）
try:
	from mymodule import confirmation_GPU

	device = confirmation_GPU.get_device()
except ImportError:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Wave_UNet.py (Original Reproduction) 使用デバイス: {device}")


class Wave_UNet(nn.Module):
	"""
	Stoller et al. (2018) の原論文を忠実に再現した Wave-U-Net。

	特徴:
	- 1D 畳み込みのみを使用 (Conv1d)
	- ダウンサンプリング: デシメーション (間引き)
	- アップサンプリング: 線形補間 (Linear Interpolation)
	- フィルタ数: 層ごとに 24 ずつ線形に増加 (24, 48, 72, ..., 288)
	- カーネルサイズ: エンコーダーは 15、デコーダーは 5
	"""

	def __init__(self, num_inputs=1, num_outputs=1, num_layers=3, initial_filter_size=24):
		super(Wave_UNet, self).__init__()
		self.num_layers = num_layers

		self.encoder_blocks = nn.ModuleList()
		self.decoder_blocks = nn.ModuleList()

		# --- エンコーダー (Downsampling Path) ---
		# 論文パラメータ: Kernel=15, Filters = i * initial_filter_size
		in_ch = num_inputs
		for i in range(num_layers):
			out_ch = initial_filter_size * (i + 1)
			self.encoder_blocks.append(
				nn.Sequential(
					nn.Conv1d(in_ch, out_ch, kernel_size=15, padding=7),  # 'same' padding
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
			in_ch = out_ch

		# --- ボトルネック (Middle Layer) ---
		# 論文パラメータ: Filters = (L+1) * initial_filter_size
		self.bottleneck = nn.Sequential(
			nn.Conv1d(in_ch, in_ch + initial_filter_size, kernel_size=15, padding=7),
			nn.LeakyReLU(0.2, inplace=True)
		)
		in_ch = in_ch + initial_filter_size

		# --- デコーダー (Upsampling Path) ---
		# 論文パラメータ: Kernel=5, フィルタ数は減少
		for i in range(num_layers - 1, -1, -1):
			skip_ch = initial_filter_size * (i + 1)
			out_ch = initial_filter_size * (i + 1)
			self.decoder_blocks.append(
				nn.Sequential(
					nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=5, padding=2),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
			in_ch = out_ch

		# --- 出力層 ---
		# 1x1 畳み込みで波形を生成
		self.out_conv = nn.Conv1d(in_ch, num_outputs, kernel_size=1)

	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): [Batch, Channels, Length] の時間波形
		"""
		skips = []

		# エンコーダー: 畳み込み + スキップ保存 + デシメーション
		for i in range(self.num_layers):
			x = self.encoder_blocks[i](x)
			skips.append(x)
			# Decimation (2サンプルごとに1つ取得)
			x = x[:, :, ::2]

		# 中間層
		x = self.bottleneck(x)

		# デコーダー: アップサンプリング + 結合 + 畳み込み
		for i in range(self.num_layers):
			# 線形補間によるアップサンプリング (倍精度)
			x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)

			# 対応する階層のスキップ結合を取り出す
			skip = skips.pop()

			# 入力長が奇数の場合に備え、サイズを合わせてパディング (境界問題への対応)
			if x.shape[-1] != skip.shape[-1]:
				x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))

			# チャンネル次元で結合
			x = torch.cat([x, skip], dim=1)
			x = self.decoder_blocks[i](x)

		# 最終出力
		x = self.out_conv(x)

		# 波形出力のため、一般的に Tanh で [-1, 1] に制限
		return torch.tanh(x)


def main():
	# パラメータ設定 (16kHz, 1.024s分の音声)
	batch = 16
	channels = 1
	length = 16000 * 12  # 2^14

	model = Wave_UNet(num_inputs=channels, num_outputs=1).to(device)

	# ダミーデータでの動作確認
	x = torch.randn(batch, channels, length).to(device)

	print("\n--- Wave-U-Net (Original Reproduction) Summary ---")
	summary(model, input_size=(batch, channels, length), device=device)

	output = model(x)
	print(f"\nInput shape:  {x.shape}")
	print(f"Output shape: {output.shape}")


if __name__ == '__main__':
	main()