#
# models/GNN.py (修正版)
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.nn import knn_graph # 不要になった
from torchinfo import summary

# --- 変更箇所 (インポート) ---
# GraphBuilder と GNN.py 内で定義されていた GraphConfig 関連を削除
# from models.graph_utils import GraphBuilder, GraphConfig, NodeSelectionType, EdgeSelectionType # 不要
from models.graph_learning_module import GraphLearningModule  # 新しくGLMをインポート

from mymodule import confirmation_GPU

device = confirmation_GPU.get_device()
print(f"GNN.py 使用デバイス: {device}")


class DoubleConv1D(nn.Module):
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

	def forward(self, x):
		return self.double_conv(x)


class Down1D(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(nn.MaxPool1d(2), DoubleConv1D(in_channels, out_channels))

	def forward(self, x):
		return self.maxpool_conv(x)


class Up1D(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv1D(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diff = x2.size()[2] - x1.size()[2]
		x1 = F.pad(x1, [diff // 2, diff - diff // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class GCN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_dim)
		self.conv3 = GCNConv(hidden_dim, output_dim)

	# --- 変更箇所 (edge_weight を受け取る) ---
	def forward(self, x, edge_index, edge_weight=None):
		x = F.relu(self.conv1(x, edge_index, edge_weight))
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu(self.conv2(x, edge_index, edge_weight))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.conv3(x, edge_index, edge_weight)
		return x


class GAT(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout_rate=0.5):
		super(GAT, self).__init__()
		self.heads = heads
		self.dropout_rate = dropout_rate

		self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
		self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_rate)
		self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout_rate)

	# --- 変更箇所 (edge_weight を受け取るが、GATConvは使わない) ---
	def forward(self, x, edge_index, edge_weight=None):
		# GATConvは自身のアテンション重みを使用するため、edge_weight は無視されます。
		# グラフ構造(edge_index)のみGLMから受け取ります。
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = F.elu(self.conv1(x, edge_index))
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = F.elu(self.conv2(x, edge_index))
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.conv3(x, edge_index)
		return x


class UGNN(nn.Module):
	def __init__(
			self,
			n_channels=1,
			n_classes=1,
			hidden_dim_gnn=32,
			# num_node=8, # GLMでは不要 (kで指定)
			gnn_type="GCN",
			gnn_heads=4,
			gnn_dropout=0.6,
			# graph_config=None, # --- 変更箇所 (削除) ---
			glm_k: int = 16,  # --- 変更箇所 (GLMのKを追加) ---
	):
		super(UGNN, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		# self.num_node_gnn = num_node # 不要
		self.gnn_type = gnn_type

		# エンコーダ
		self.encoder_dim = 512
		self.sampling_rate = 16000
		self.win = 4
		self.win_samples = int(self.sampling_rate * self.win / 1000)
		self.stride_samples = self.win_samples // 2

		self.encoder = nn.Conv1d(
			in_channels=n_channels,
			out_channels=self.encoder_dim,
			kernel_size=self.win_samples,
			bias=False,
			stride=self.stride_samples,
		)

		# U-Net (1D)
		self.inc = DoubleConv1D(self.encoder_dim, 64)
		self.down1 = Down1D(64, 128)
		self.down2 = Down1D(128, 256)
		self.down3 = Down1D(256, 512)

		# ボトルネックのGNN
		gnn_input_dim = 512  # ボトルネックの特徴量次元
		if gnn_type == "GCN":
			self.gnn = GCN(gnn_input_dim, hidden_dim_gnn, gnn_input_dim)
		elif gnn_type == "GAT":
			self.gnn = GAT(gnn_input_dim, hidden_dim_gnn, gnn_input_dim, heads=gnn_heads, dropout_rate=gnn_dropout)

		# --- 変更箇所 (GraphBuilder -> GLM) ---
		# self.graph_builder = GraphBuilder(graph_config)
		self.glm = GraphLearningModule(
			input_dim=gnn_input_dim,
			k=glm_k
		)
		# グラフ正則化損失をトレーナーに渡すためのプレースホルダ
		self.latest_graph_reg_loss = torch.tensor(0.0, device=device)
		# --- 変更ここまで ---

		# デコーダパス
		self.up1 = Up1D(512, 256)
		self.up2 = Up1D(256, 128)
		self.up3 = Up1D(128, 64)
		self.outc = nn.Sequential(nn.Conv1d(64, n_classes, kernel_size=1), nn.Sigmoid())

		# デコーダ
		self.decoder = nn.ConvTranspose1d(
			in_channels=self.encoder_dim,
			out_channels=n_classes,  # マスク適用後のため、encoder_dim -> n_classes
			kernel_size=self.win_samples,
			bias=False,
			stride=self.stride_samples,
		)

	# --- graph_config のデフォルト設定 (削除) ---
	# if graph_config is None: ...
	# self.graph_builder = GraphBuilder(graph_config)

	def forward(self, x):
		# エンコーダ
		x_encoded = self.encoder(x)

		# U-Net エンコーダパス
		x1 = self.inc(x_encoded)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)

		# ボトルネックのGNN処理
		batch_size, channels, length = x4.size()
		x4_nodes = x4.permute(0, 2, 1).reshape(-1, channels)

		# --- 変更箇所 (GraphBuilder -> GLM) ---
		# k-NNグラフの作成
		# edge_index = self.graph_builder.create_batch_graph(x4_nodes, batch_size, length)

		# GLMによる動的グラフ生成
		edge_index, edge_weight, graph_reg_loss = self.glm(
			x=x4_nodes,
			batch_size=batch_size,
			nodes_per_sample=length
		)
		# 損失を保存
		self.latest_graph_reg_loss = graph_reg_loss
		# --- 変更ここまで ---

		# GNN処理
		x4_processed = self.gnn(x4_nodes, edge_index, edge_weight=edge_weight)
		x4_processed = x4_processed.view(batch_size, length, channels).permute(0, 2, 1)

		# U-Net デコーダパス
		x = self.up1(x4_processed, x3)
		x = self.up2(x, x2)
		x = self.up3(x, x1)

		# マスク生成
		mask = self.outc(x)

		# マスク適用
		masked_features = x_encoded * mask

		# デコーダで波形に変換
		out = self.decoder(masked_features)

		return out


def print_model_summary(model, batch_size, channels, length):
	# サンプル入力データを作成
	x = torch.randn(batch_size, channels, length).to(device)

	# モデルのサマリーを表示
	# --- 変更箇所 (グラフ設定の表示を削除) ---
	print(f"\n--- {model.gnn_type} with Dynamic GraphLearningModule (GLM) ---")
	summary(model, input_data=x)


def main():
	print("GNN.py main execution")
	# サンプルデータの作成（入力サイズを縮小）
	batch = 1
	num_mic = 1
	length = 16000 * 8  # 8秒の音声データ (例)
	glm_k_test = 16  # GLMのTop-K

	gnn_types = ["GCN", "GAT"]

	# --- 変更箇所 (graph_config のループを削除) ---
	# node_selection_types = [NodeSelectionType.ALL, NodeSelectionType.TEMPORAL]
	# edge_selection_types = [EdgeSelectionType.RANDOM, EdgeSelectionType.KNN]

	for gnn_type in gnn_types:
		# for node_selection in node_selection_types:
		#     for edge_selection in edge_selection_types:
		#
		#         graph_config = GraphConfig(
		#             num_edges=num_node,
		#             node_selection=node_selection,
		#             edge_selection=edge_selection,
		#         )

		model = UGNN(
			n_channels=num_mic,
			n_classes=1,
			# num_node=num_node,
			gnn_type=gnn_type,
			# graph_config=graph_config, # 削除
			glm_k=glm_k_test  # 追加
		).to(device)

		model_type = f"{gnn_type}_DynamicGLM_k{glm_k_test}"
		print(f"\nModel Type: {model_type}")
		print_model_summary(model, batch, num_mic, length)

		# メモリ使用量の表示
		if torch.cuda.is_available():
			print(f"\nGPU Memory Usage (after initializations):")
			print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
			print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
	# モデルのテスト
	main()