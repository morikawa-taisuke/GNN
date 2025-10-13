from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torchinfo import summary

from models.graph_utils import GraphBuilder, GraphConfig, NodeSelectionType, EdgeSelectionType
from mymodule import confirmation_GPU

device = confirmation_GPU.get_device()
print(f"CheckSpeqGNN.py 使用デバイス: {device}")


# --- 共通ブロックの定義 (models/SpeqGNN.py より) ---

class DoubleConv(nn.Module):
	"""(畳み込み => バッチ正規化 => ReLU) を2回繰り返すブロック"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
	"""ダウンサンプリングブロック (マックスプーリング + DoubleConv)"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

	def forward(self, x): return self.maxpool_conv(x)


class Up(nn.Module):
	"""アップサンプリングブロック (転置畳み込み + DoubleConv)"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
		                diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class GCN(nn.Module):
	"""3層のGraph Convolutional Network (GCN) モデル"""

	def __init__(self, input_dim, hidden_dim, output_dim):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_dim)
		self.conv3 = GCNConv(hidden_dim, output_dim)

	def forward(self, x, edge_index):
		x = F.relu(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu(self.conv2(x, edge_index))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.conv3(x, edge_index)
		return x


class GAT(nn.Module):
	"""3層のGraph Attention Network (GAT) モデル"""

	def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout_rate=0.5):
		super(GAT, self).__init__()
		self.heads = heads
		self.dropout_rate = dropout_rate
		self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
		self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_rate)
		self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout_rate)

	def forward(self, x, edge_index):
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = F.elu(self.conv1(x, edge_index))
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = F.elu(self.conv2(x, edge_index))
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.conv3(x, edge_index)
		return x


# --- 検証用モデルの定義 ---
class CheckSpeqGNN(nn.Module):
	"""
	SpeqGNNをベースに、ノードごとの誤差とエッジ情報を出力する検証用モデル。
	"""

	def __init__(self, n_channels=1, n_classes=1, *,
	             gnn_type="GAT",
	             graph_config: GraphConfig,
	             hidden_dim=32,
	             gat_heads=4,
	             gat_dropout=0.6,
	             n_fft=512,
	             hop_length=256,
	             win_length=None,
	             device=device):

		super(CheckSpeqGNN, self).__init__()

		self.device = device
		# モデルの基本的な層はSpeqGNNと共通
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.gnn_type = gnn_type
		self.graph_builder = GraphBuilder(graph_config)

		# ISTFT用のパラメータと変換器
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.win_length = win_length if win_length is not None else n_fft
		self.window = torch.hann_window(self.win_length)

		# U-Net エンコーダ部分 (教師データを流すため、同じ層を定義)
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)

		# ボトルネック部分のGNN
		if gnn_type.upper() == "GCN":
			self.gnn = GCN(512, hidden_dim, 512)
		elif gnn_type.upper() == "GAT":
			self.gnn = GAT(512, hidden_dim, 512, heads=gat_heads, dropout_rate=gat_dropout)
		else:
			raise ValueError(f"Unsupported GNN type: {gnn_type}")

		# U-Net デコーダ部分
		self.up1 = Up(512, 256)
		self.up2 = Up(256, 128)
		self.up3 = Up(128, 64)
		self.outc = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

	def _get_encoder_features(self, magnitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		振幅スペクトログラムを入力として、U-Netエンコーダの特徴マップとボトルネック特徴マップを返す。
		"""
		x1 = self.inc(magnitude)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)  # ボトルネック特徴量
		return x1, x2, x3, x4

	def forward(self,
	            noisy_magnitude: torch.Tensor,
	            clean_magnitude: torch.Tensor,  # ★追加: 教師信号の振幅スペクトログラム
	            complex_spec_input: torch.Tensor,
	            original_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		順伝播を拡張し、ノードごとの誤差とエッジインデックスを返す。

		Args:
			noisy_magnitude (torch.Tensor): ノイズ入り信号の振幅スペクトログラム [B, C, F, T]
			clean_magnitude (torch.Tensor): クリーン信号の振幅スペクトログラム [B, C, F, T]
			complex_spec_input (torch.Tensor): ノイズ入り信号の複素スペクトログラム [B, F, T]
			original_length (int, optional): 元の波形データの長さ。

		Returns:
			Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
				(強調後の音声波形, ノードごとの平均誤差, 構築されたエッジインデックス)
		"""
		# print(noisy_magnitude.shape)
		batch_size, _, input_freq_bins, input_time_frames = noisy_magnitude.size()

		# --- 1. エンコーダ特徴量の取得 ---
		# ノイズあり信号のエンコーダパス
		x1_noisy, x2_noisy, x3_noisy, x4_noisy = self._get_encoder_features(noisy_magnitude)

		# クリーン信号のエンコーダパス (同じ重みでフォワード)
		with torch.no_grad():  # 教師信号のパスは勾配計算から除外
			_, _, _, y4_clean = self._get_encoder_features(clean_magnitude.to(self.device))

		# --- 2. ノードごとの誤差計算 (ノイズ vs クリーン) ---
		_, channels_bottleneck, height_bottleneck, width_bottleneck = x4_noisy.size()

		# ノード次元にリシェイプ [B * N_nodes, C]
		x4_reshaped_noisy = x4_noisy.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)
		y4_reshaped_clean = y4_clean.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)

		# ノードごとのL2誤差 (MSE) を計算し、ノードごとに平均化 [B * N_nodes]
		# dim=1で特徴次元（512）の平均誤差を計算
		node_loss = torch.mean((x4_reshaped_noisy - y4_reshaped_clean) ** 2, dim=1)

		# --- 3. GNNの実行とエッジの取得 ---
		# GraphBuilderを使用してグラフを作成
		num_nodes_per_sample = height_bottleneck * width_bottleneck
		edge_index = self.graph_builder.create_batch_graph(x=x4_reshaped_noisy,
		                                                   batch_size=batch_size,
		                                                   nodes_per_sample=num_nodes_per_sample)

		# GNNによるノード特徴の更新
		x4_processed_flat = self.gnn(x4_reshaped_noisy, edge_index)

		# GNNの出力をU-Netの形状に戻す
		x4_processed = x4_processed_flat.view(batch_size, height_bottleneck, width_bottleneck,
		                                      channels_bottleneck).permute(0, 3, 1, 2)

		# --- 4. U-Net デコーダとマスクの適用 (SpeqGNN.pyの残りのロジック) ---
		d3 = self.up1(x4_processed, x3_noisy)
		d2 = self.up2(d3, x2_noisy)
		d1 = self.up3(d2, x1_noisy)

		mask_pred_raw = self.outc(d1)
		mask_pred = torch.sigmoid(mask_pred_raw)

		# 5. マスクを入力特徴マップのサイズにリサイズ
		if mask_pred.size(2) != input_freq_bins or mask_pred.size(3) != input_time_frames:
			mask_pred_resized = F.interpolate(mask_pred, size=(input_freq_bins, input_time_frames), mode='bilinear',
		                                      align_corners=False)
		else:
			mask_pred_resized = mask_pred

		# マスクの適用
		predicted_magnitude_tf = noisy_magnitude * mask_pred.squeeze(1)

		# ISTFTによる波形再構成
		# ... (ISTFTロジック: 割愛)
		phase = torch.angle(complex_spec_input)
		reconstructed_complex_spec = torch.polar(predicted_magnitude_tf.squeeze(1), phase)

		output_waveform = torchaudio.transforms.InverseSpectrogram(
			n_fft=self.n_fft,
			hop_length=self.hop_length,
			win_length=self.win_length,
			window_fn=lambda n: torch.hann_window(n).to(reconstructed_complex_spec.device)
		).to(self.device)(reconstructed_complex_spec, length=original_length)

		# 3つの重要な情報を返す
		return output_waveform, node_loss, edge_index


def print_model_summary(model: CheckSpeqGNN, batch_size, length):
	"""検証モデルのサマリーを表示するヘルパー関数"""
	n_fft = model.n_fft
	hop_length = model.hop_length
	win_length = model.win_length
	window = torch.hann_window(win_length, device=device)

	# サンプル入力データの作成
	x_time = torch.randn(batch_size, 1, length, device=device)
	y_time = torch.randn(batch_size, 1, length, device=device)  # 教師波形

	# --- 入力データをSTFTでスペクトログラムに変換 ---
	to_spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
	                                            window_fn=lambda n: torch.hann_window(n).to(device), return_complex=True).to(
		device)

	x_complex_spec = to_spec(x_time.squeeze(1))
	x_magnitude_spec = torch.abs(x_complex_spec).unsqueeze(1)

	y_complex_spec = to_spec(y_time.squeeze(1))
	y_magnitude_spec = torch.abs(y_complex_spec).unsqueeze(1)

	original_len = x_time.shape[-1]

	# モデルのサマリーを表示
	print(f"\n--- {model.gnn_type} with {model.graph_builder.config.edge_selection.value} Edges ---")
	summary(model, input_data=(x_magnitude_spec, y_magnitude_spec, x_complex_spec, original_len), device=device)


if __name__ == '__main__':
	# --- メイン実行 (モデル構造の確認) ---
	print("CheckSpeqGNN のメイン実行（モデルテスト用）")

	# パラメータ設定
	batch = 1
	length = 16000 * 4  # 4秒の音声データ (例)
	num_node_edges = 8

	# GraphConfig (UGAT + KNNを例として設定)
	graph_config = GraphConfig(
		num_edges=num_node_edges,
		node_selection=NodeSelectionType.ALL,
		edge_selection=EdgeSelectionType.KNN,
		bidirectional=True,
	)

	# モデルのインスタンス化
	model = CheckSpeqGNN(
		n_channels=1,
		n_classes=1,
		gnn_type="GAT",
		graph_config=graph_config,
		gat_heads=4,
		gat_dropout=0.6,
		n_fft=512,
		hop_length=256,
		win_length=512
	).to(device)

	# モデルのサマリー表示 (引数が4つあることに注意)
	print_model_summary(model, batch, length)

	# サンプルデータでのフォワードパス実行
	x_mag = torch.randn(batch, 1, 257, 126).to(device)  # F=257, T=126 (4秒音声, n_fft=512で計算)
	y_mag = torch.randn(batch, 1, 257, 126).to(device)
	x_comp = torch.randn(batch, 257, 126, dtype=torch.complex64).to(device)

	output_w, node_loss, edge_idx = model(x_mag, y_mag, x_comp, length)

	print(f"\n--- 拡張された出力の形状 ---")
	print(f"1. 強調後の音声波形: {output_w.shape}")
	print(f"2. ノードごとの平均誤差 (平坦化): {node_loss.shape}")
	print(f"3. 構築されたエッジインデックス: {edge_idx.shape}")

	# ノード数（H*W）の確認
	N_nodes = node_loss.shape[0]
	print(f"\n検証: ボトルネックノード数: {N_nodes}")
