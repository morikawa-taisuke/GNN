from __future__ import print_function

import time  # 時間
from librosa.core import stft, istft
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm.contrib import tenumerate
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from itertools import permutations
from torch.nn.utils import weight_norm
import scipy.signal as sp
import scipy as scipy
from torchinfo import summary
import os
from pathlib import Path

from mymodule import const, confirmation_GPU

# CUDAのメモリ管理設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAの可用性をチェック
device = confirmation_GPU.get_device()
print(f"wave_unet.py 使用デバイス: {device}")


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


class conv_block(nn.Module):
	def __init__(self, ch_in, ch_out):
		super(conv_block, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x


class up_conv(nn.Module):
	def __init__(self, ch_in, ch_out):
		super(up_conv, self).__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.up(x)
		return x


class U_Net(nn.Module):
	def __init__(self, input_ch=1, output_ch=1):
		super(U_Net, self).__init__()
		self.encoder_dim = 512
		self.sampling_rate = 16000
		self.win = 4
		self.win = int(self.sampling_rate * self.win / 1000)
		self.stride = self.win // 2  # 畳み込み処理におけるフィルタが移動する幅

		self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

		# self.encoder = nn.Conv1d(in_channels=input_ch, out_channels=)
		self.encoder = nn.Conv1d(in_channels=input_ch,  # 入力データの次元数 #=1もともとのやつ
		                         out_channels=self.encoder_dim,  # 出力データの次元数
		                         kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長なの?)
		                         bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
		                         stride=self.stride)  # 畳み込み処理の移動幅

		self.Conv1 = conv_block(ch_in=input_ch, ch_out=64)
		self.Conv2 = conv_block(ch_in=64, ch_out=128)
		self.Conv3 = conv_block(ch_in=128, ch_out=256)

		self.Up3 = up_conv(ch_in=256, ch_out=128)
		self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

		self.Up2 = up_conv(ch_in=128, ch_out=64)
		self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

		self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

		self.decoder = nn.ConvTranspose1d(in_channels=self.encoder_dim,  # 入力次元数
		                                  out_channels=output_ch,  # 出力次元数 1もともとのやつ
		                                  kernel_size=self.win,  # カーネルサイズ
		                                  bias=False,
		                                  stride=self.stride)  # 畳み込み処理の移動幅

	def forward(self, x):
		# encoding path
		# print("x: ", x.shape)
		x = self.encoder(x)
		# print("encoder out: ", x.shape)
		# x = x.unsqueeze(dim=0)
		x = x.unsqueeze(dim=1)
		x1 = self.Conv1(x)

		x2 = self.Maxpool(x1)
		x2 = self.Conv2(x2)

		x3 = self.Maxpool(x2)
		x3 = self.Conv3(x3)

		d3 = self.Up3(x3)
		x2, d3 = padding_tensor(x2, d3)
		d3 = torch.cat((x2, d3), dim=1)
		d3 = self.Up_conv3(d3)

		d2 = self.Up2(d3)
		x1, d2 = padding_tensor(x1, d2)
		d2 = torch.cat((x1, d2), dim=1)
		d2 = self.Up_conv2(d2)

		d1 = self.Conv_1x1(d2)
		d1 = torch.sigmoid(d1)
		out = x * d1
		# print("out: ", out.shape)
		out = out.squeeze()
		out = self.decoder(out)

		return out


def print_model_summary(model, batch, num_mic, length):
	# サンプル入力データを作成
	x = torch.randn(batch, num_mic, length).to(device)

	# モデルのサマリーを表示
	print("\nURelNet Model Summary:")
	summary(model, input_data=x)


def model_check():
	print("main")
	# サンプルデータの作成（入力サイズを縮小）
	batch = const.BATCHSIZE
	num_mic = 1  # 入力サイズを縮小
	length = 128000  # 入力サイズを縮小

	# ランダムな入力画像を作成
	x = torch.randn(batch, num_mic, length).to(device)

	# モデルの初期化とデバイスへの移動
	model = U_Net().to(device)

	# モデルのサマリーを表示
	print_model_summary(model, batch, num_mic, length)

	# フォワードパス
	output = model(x)
	print(f"\nInput shape: {x.shape}")
	print(f"Output shape: {output.shape}")

	# メモリ使用量の表示
	if torch.cuda.is_available():
		print(f"\nGPU Memory Usage:")
		print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
		print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == '__main__':
	model_check()