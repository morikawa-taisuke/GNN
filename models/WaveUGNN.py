import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torchinfo import summary

# 既存のユーティリティを利用
from models.graph_utils import GraphBuilder, GraphConfig, NodeSelectionType, EdgeSelectionType
from mymodule import confirmation_GPU

device = confirmation_GPU.get_device()


class GNN_Bottleneck(nn.Module):
	""" 3層のGNN構成 (GCN または GAT) """

	def __init__(self, input_dim, hidden_dim, output_dim, gnn_type="GCN", heads=4, dropout=0.2):
		super().__init__()
		self.gnn_type = gnn_type
		self.dropout = dropout

		if gnn_type == "GCN":
			self.conv1 = GCNConv(input_dim, hidden_dim)
			self.conv2 = GCNConv(hidden_dim, hidden_dim)
			self.conv3 = GCNConv(hidden_dim, output_dim)

		elif gnn_type == "GAT":
			# GATの実装 (GNN.py の構造を参考)
			self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
			# 2層目は 1層目の出力(hidden_dim * heads)を入力とする
			self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
			# 最終層は出力を結合せず平均化(concat=False)して元の次元に戻す
			self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)

	def forward(self, x, edge_index):
		if self.gnn_type == "GCN":
			x = F.relu(self.conv1(x, edge_index))
			x = F.relu(self.conv2(x, edge_index))
			return self.conv3(x, edge_index)

		elif self.gnn_type == "GAT":
			# GATでは一般的に ELU 活性化関数と Dropout が使われる (GNN.py 参照)
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = F.elu(self.conv1(x, edge_index))
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = F.elu(self.conv2(x, edge_index))
			x = F.dropout(x, p=self.dropout, training=self.training)
			return self.conv3(x, edge_index)


class Wave_UGNN(nn.Module):
	"""
	原論文再現版 Wave-U-Net + GNNボトルネック
	"""

	def __init__(self, num_inputs=1, num_outputs=1, num_layers=12, initial_filter_size=24, gnn_type="GCN", graph_config=None):
		super(Wave_UGNN, self).__init__()
		self.num_layers = num_layers

		self.encoder_blocks = nn.ModuleList()
		self.decoder_blocks = nn.ModuleList()

		# --- エンコーダー (原論文再現: Kernel=15, LeakyReLU) ---
		in_ch = num_inputs
		for i in range(num_layers):
			out_ch = initial_filter_size * (i + 1)
			self.encoder_blocks.append(
				nn.Sequential(
					nn.Conv1d(in_ch, out_ch, kernel_size=15, padding=7),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
			in_ch = out_ch

		# --- ボトルネック (ここをGNNに置換) ---
		self.bottleneck_dim = in_ch
		self.gnn = GNN_Bottleneck(self.bottleneck_dim, 256, self.bottleneck_dim, gnn_type=gnn_type)

		# グラフ構築用
		self.graph_builder = GraphBuilder(graph_config)


		# --- デコーダー (原論文再現: Kernel=5) ---
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

		self.out_conv = nn.Conv1d(in_ch, num_outputs, kernel_size=1)

	def forward(self, x):
		skips = []

		# print(x.shape)
		# exit()
		# エンコーダー: 畳み込み + デシメーション
		for i in range(self.num_layers):
			x = self.encoder_blocks[i](x)
			skips.append(x)
			x = x[:, :, ::2]  # Decimation
		# print(x.shape)
		# --- GNN ボトルネック処理 ---
		x = x.unsqueeze(1)  # [B, C, L] -> [B, 1, C, L]   3次元から4次元に変換 (1) 非対応
		# x = x.unsqueeze(2)  # [B, C, L] -> [B, C, 1, L] 3次元から4次元に変換   (2)
		# print(x.shape)
		batch_size, num_mic, channels, length = x.size()  # 形状の取得

		# x_nodes = x.permute(0, 2, 1).reshape(-1, channels)  # [B*L, C]
		x_nodes = x.view(batch_size, num_mic, -1).permute(0, 2, 1).reshape(-1, num_mic)   # ノード用にリシェイプ
		# print(x_nodes.shape)
		edge_index = self.graph_builder.create_batch_graph(x_features_4d=x) # グラフ構築 (入力は4次元の特徴量)
		# print(edge_index.shape)
		x_gnn = self.gnn(x_nodes, edge_index)   # GNN処理

		# x = x_gnn.view(batch_size, length, channels).permute(0, 2, 1)   # 元の形状に戻す
		x = x_gnn.view(batch_size, channels, length, num_mic).permute(0, 3, 1, 2)  # 元の形状に戻す
		# print("||||||||||||||||||||||||||"*50)
		# print(x.shape)
		# exit()
		x = x.squeeze(1)  # [B, 1, C, L] -> [B, C, L]   4次元から3次元に変換 (1) 非対応
		# x = x.squeeze(2)  # [B, C, 1, L] -> [B, C, L]   4次元から3次元に変換 (2)

		# デコーダー: 線形補間アップサンプリング + スキップ結合
		for i in range(self.num_layers):
			x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
			skip = skips.pop()

			if x.shape[-1] != skip.shape[-1]:
				x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))

			x = torch.cat([x, skip], dim=1)
			x = self.decoder_blocks[i](x)

		return torch.tanh(self.out_conv(x))


def main():
	model = Wave_UGNN().to(device)
	summary(model, input_size=(1, 1, 16384), device=device)


if __name__ == '__main__':
	main()