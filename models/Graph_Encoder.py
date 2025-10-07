import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torchinfo import summary  # モデルのサマリー表示用

# 必要なユーティリティはインポートで対応
from models.graph_utils import GraphBuilder, GraphConfig, NodeSelectionType, EdgeSelectionType
from mymodule import confirmation_GPU

# PyTorchのCUDAメモリ管理設定。セグメントを拡張可能にすることで、断片化によるメモリ不足エラーを緩和します。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CUDAが利用可能かチェックし、利用可能ならGPUを、そうでなければCPUを使用するデバイスとして設定します。
device = confirmation_GPU.get_device()
print(f"GNN_ReverbFeatureEncoder.py 使用デバイス: {device}")


# --- 1. ユーティリティークラス (既存ファイルからの再利用) ---

class DoubleConv(nn.Module):
	"""(畳み込み => バッチ正規化 => ReLU) を2回繰り返すブロック (1D Conv)"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
	"""ダウンサンプリングブロック (マックスプーリング + DoubleConv)"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(nn.MaxPool1d(2), DoubleConv(in_channels, out_channels))

	def forward(self, x): return self.maxpool_conv(x)


class Up(nn.Module):
	"""アップサンプリングブロック"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diff = x2.size(2) - x1.size(2)
		x1 = F.pad(x1, [diff // 2, diff - diff // 2])
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


# --- 2. Reverb Feature Encoder (補助タスク) ---
class ReverbFeatureEncoder(nn.Module):
	"""
	時間-周波数表現（1D Conv出力）から残響特性を抽出する補助エンコーダ (CNN-LSTMベース)。
	"""

	def __init__(self, input_dim, hidden_dim_cnn=64, hidden_dim_lstm=64, feature_dim=19):
		super(ReverbFeatureEncoder, self).__init__()

		# 1. CNN層: 空間/チャネル次元の次元削減と非線形化 (ConvTasNetのアイデアを応用)
		self.conv1x1 = nn.Sequential(
			nn.Conv1d(input_dim, hidden_dim_cnn, kernel_size=1),
			nn.GroupNorm(1, hidden_dim_cnn, eps=1e-8),
			nn.PReLU()
		)

		# 2. LSTM層: 時間的コンテキストの集約 (残響の減衰特性を捉える)
		self.lstm = nn.LSTM(
			input_size=hidden_dim_cnn,
			hidden_size=hidden_dim_lstm,
			num_layers=1,
			batch_first=True,
			bidirectional=False
		)

		# 3. FC層: 最終的な残響特徴量の出力
		self.fc_out = nn.Sequential(
			nn.Linear(hidden_dim_lstm, hidden_dim_lstm // 2),
			nn.PReLU(),
			nn.Linear(hidden_dim_lstm // 2, feature_dim)  # 最終的な残響特徴量次元
		)

	def forward(self, x):
		# x: [B, C, L] (例: B, 512, L_encoded)

		# 1. 1x1 Convで次元削減: [B, 512, L] -> [B, hidden_dim_cnn, L]
		x = self.conv1x1(x)

		# 2. LSTM入力形式に変換: [B, C, L] -> [B, L, C]
		x_permuted = x.permute(0, 2, 1)

		# LSTM処理
		out, _ = self.lstm(x_permuted)

		# 3. 時間軸での平均プーリング (シーケンス全体を一つのベクトルに集約)
		reverb_vector = out.mean(dim=1)

		# 4. FC層で最終的な残響特徴量を出力
		reverb_feature = self.fc_out(reverb_vector)

		return reverb_feature

# --- 3. メインモデル: ReverbGNNEncoder (マルチタスク出力) ---
class ReverbGNNEncoder(nn.Module):
	"""
	残響特徴量エンコーダの出力をGNNの入力特徴量として利用し、
	強調音声と学習済み残響特徴量の両方を出力するマルチタスクモデル。
	"""

	def __init__(
			self,
			n_channels=1,
			n_classes=1,
			hidden_dim_gnn=32,
			num_node=8,
			gnn_type="GCN",
			gnn_heads=4,
			gnn_dropout=0.6,
			graph_config=None,
			reverb_feature_dim=19  # 補助タスクの教師信号次元 (ケプストラム係数+スカラー特徴)
	):
		super(ReverbGNNEncoder, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.num_node_gnn = num_node
		self.gnn_type = gnn_type
		self.reverb_feature_dim = reverb_feature_dim

		# 1. 初期1D畳み込みエンコーダ (波形 -> 潜在特徴量)
		self.encoder_dim_1d_conv = 512  # GNNEncoder.pyと同じ
		self.sampling_rate = 16000
		self.win = 4
		self.win_samples = int(self.sampling_rate * self.win / 1000)
		self.stride_samples = self.win_samples // 2

		self.initial_encoder = nn.Conv1d(
			in_channels=self.n_channels,
			out_channels=self.encoder_dim_1d_conv,
			kernel_size=self.win_samples,
			bias=False,
			stride=self.stride_samples,
		)

		# 2. Reverb Feature Encoder (補助タスク)
		self.reverb_feature_encoder = ReverbFeatureEncoder(
			input_dim=self.encoder_dim_1d_conv,
			feature_dim=self.reverb_feature_dim
		)

		# 3. GNN層
		# GNNの入力次元は、Reverb Feature Encoderの出力次元に合わせる
		gnn_input_dim = self.reverb_feature_dim
		# GNNの出力次元は、U-Netの入力次元（元の潜在特徴量次元）に戻す
		gnn_output_dim = self.encoder_dim_1d_conv

		if gnn_type == "GCN":
			self.gnn_encoder = GCN(gnn_input_dim, hidden_dim_gnn, gnn_output_dim)
		elif gnn_type == "GAT":
			self.gnn_encoder = GAT(
				gnn_input_dim,
				hidden_dim_gnn,
				gnn_output_dim,
				heads=gnn_heads,
				dropout_rate=gnn_dropout,
			)
		else:
			raise ValueError(f"Unsupported GNN type: {gnn_type}。'GCN' または 'GAT' を選択してください。")

		# 4. U-Net構造 (GNNの出力が入力)
		self.inc = DoubleConv(self.encoder_dim_1d_conv, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)

		self.up1 = Up(512, 256)
		self.up2 = Up(256, 128)
		self.up3 = Up(128, 64)

		# マスク用の出力畳み込み
		self.outc = nn.Sequential(nn.Conv1d(64, n_classes, kernel_size=1), nn.Sigmoid())

		# 5. 最終的な1D転置畳み込みデコーダ
		self.final_decoder = nn.ConvTranspose1d(
			in_channels=self.encoder_dim_1d_conv,
			out_channels=self.n_channels,
			kernel_size=self.win_samples,
			bias=False,
			stride=self.stride_samples,
		)
		# 6. グラフ設定
		if graph_config is None:
			graph_config = GraphConfig(
				num_edges=num_node,
				node_selection=NodeSelectionType.ALL,
				edge_selection=EdgeSelectionType.KNN,
				bidirectional=True,
			)
		self.graph_builder = GraphBuilder(graph_config)

	def forward(self, x_waveform):
		"""
		順伝播 (マルチタスク出力)

		Returns:
			output_waveform (torch.Tensor): 強調後の音声波形 [B, C, L]
			reverb_feature (torch.Tensor): 学習済み残響特徴量 [B, feature_dim]
		"""
		if x_waveform.dim() == 2:
			x_waveform = x_waveform.unsqueeze(1)

		original_waveform_length = x_waveform.size(-1)

		# 1. 初期1D畳み込みエンコーダ (波形 -> 潜在特徴量)
		# x_encoded: [B, 512, L_encoded]
		x_encoded = self.initial_encoder(x_waveform)

		# 2. Reverb Feature Encoder (補助タスク)
		# reverb_feature: [B, feature_dim=19]
		reverb_feature = self.reverb_feature_encoder(x_encoded)

		batch_size, feature_dim_encoded, length_encoded = x_encoded.size()

		# 3. GNNノードの特徴量準備
		# GNNのノード特徴量として、ReverbFeatureEncoderの出力を各時間ステップに複製して利用する
		# x_nodes_for_gnn: [B * L_encoded, D_reverb]
		x_nodes_for_gnn = reverb_feature.unsqueeze(1).repeat(1, length_encoded, 1).reshape(-1, self.reverb_feature_dim)

		# GNN用のグラフを作成
		num_nodes_per_sample = length_encoded
		if num_nodes_per_sample > 0:
			# ★注意: graph_builderのKNNロジック内で、この x_nodes_for_gnn が距離計算に使われる必要があります。
			edge_index = self.graph_builder.create_batch_graph(x_nodes_for_gnn, batch_size, length_encoded)
		else:
			edge_index = torch.empty((2, 0), dtype=torch.long, device=x_nodes_for_gnn.device)

		# 4. GNNエンコーダ (ノード特徴量の処理)
		# x_gnn_out_flat: [B * L_encoded, gnn_output_dim=512]
		x_gnn_out_flat = self.gnn_encoder(x_nodes_for_gnn, edge_index)

		# GNNの出力をU-Net用の特徴マップ形式にリシェイプ
		x_gnn_out_reshaped = x_gnn_out_flat.view(batch_size, length_encoded, feature_dim_encoded).permute(0, 2, 1)

		# 5. U-Netエンコーダ/デコーダパス (入力はGNNの出力)
		x1 = self.inc(x_gnn_out_reshaped)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)

		d3 = self.up1(x4, x3)
		d2 = self.up2(d3, x2)
		d1 = self.up3(d2, x1)

		# 6. マスクの生成と適用
		mask_pred_raw = self.outc(d1)
		mask_target_W = length_encoded

		if mask_pred_raw.size(2) != length_encoded:
			mask_resized = F.interpolate(mask_pred_raw, size=length_encoded, mode="linear", align_corners=False)
		else:
			mask_resized = mask_pred_raw

		# マスクをx_encodedに適用
		if self.n_classes == 1:
			masked_features = x_encoded * mask_resized
		else:
			print(f"警告: ReverbGNNEncoder - n_classes > 1 ({self.n_classes})。最初のマスクチャネルを使用します。")
			masked_features = x_encoded * mask_resized[:, 0, :, :]

		# 7. 最終的な1D畳み込みデコーダ (マスクされた潜在特徴量 -> 波形)
		output_waveform = self.final_decoder(masked_features)

		# 出力波形を元の入力長に合わせるためにトリミング
		output_waveform = output_waveform[:, :, :original_waveform_length]

		# ★マルチタスク出力: 強調音声と学習済み残響特徴量を返す
		return output_waveform, reverb_feature