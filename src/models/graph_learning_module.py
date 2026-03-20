#
# models/graph_learning_module.py
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj


class GraphLearningModule(nn.Module):
	"""
	エンドツーエンドのグラフ構造学習モジュール (GLM)

	Document/アイデア2_E2Eのグラフ構造学習.md に基づく実装。
	U-Netのボトルネック特徴量 X から、学習可能な隣接行列 A と
	グラフ正則化損失 L_reg を計算する。
	"""

	def __init__(self,
	             input_dim: int,
	             k: int,
	             embedding_dim: int = None,
	             leaky_relu_slope: float = 0.2):
		"""
		Args:
			input_dim (int): 入力ノード特徴量 X の次元 (例: 512)
			k (int): 各ノードが接続するエッジの数 (Top-K の K)
			embedding_dim (int): Q, K の埋め込み次元。Noneの場合、input_dim と同じ。
			leaky_relu_slope (float): LeakyReLU の負の傾き
		"""
		super().__init__()

		if embedding_dim is None:
			embedding_dim = input_dim

		self.input_dim = input_dim
		self.embedding_dim = embedding_dim
		self.k = k

		# 1. 特徴量の変換 (Q, K の生成用)
		# Document/アイデア2 の「特徴量の変換」に対応
		self.query_transform = nn.Linear(input_dim, embedding_dim)
		self.key_transform = nn.Linear(input_dim, embedding_dim)

		self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

	def forward(self,
	            x: torch.Tensor,
	            batch_size: int,
	            nodes_per_sample: int):
		"""
		順伝播。

		Args:
			x (torch.Tensor): バッチ化されたノード特徴量
							  [B * N, C] (B=batch_size, N=nodes_per_sample, C=input_dim)
			batch_size (int): バッチサイズ B
			nodes_per_sample (int): 1サンプルあたりのノード数 N

		Returns:
			edge_index (torch.Tensor): 動的に生成されたグラフ接続 [2, E_total]
			edge_weight (torch.Tensor): 動的に生成されたエッジ重み [E_total]
			graph_reg_loss (torch.Tensor): グラフ正則化損失 (L_reg)
		"""

		# x を [B, N, C] にリシェイプ
		x_reshaped = x.view(batch_size, nodes_per_sample, -1)

		# 2. 関連度スコアの計算 (Q, K)
		# S = LeakyReLU(QK^T)
		#
		query = self.query_transform(x_reshaped)  # [B, N, C_emb]
		key = self.key_transform(x_reshaped)  # [B, N, C_emb]

		# S: 関連度スコア行列 [B, N, N]
		scores = self.leaky_relu(torch.bmm(query, key.transpose(1, 2)))

		# 3. 微分可能な疎結合の誘導 (Soft Top-K フィルタリング)
		# ここでは、微分可能性を維持しつつ疎結合を実現するため、
		# スコア行列 S から直接 Top-K を選択し、
		# 選択されたエッジのスコアを正規化して隣接行列 A を作成する。
		#

		# k はノード数 N を超えられない
		k_safe = min(self.k, nodes_per_sample)

		# top_k_values, top_k_indices: [B, N, k_safe]
		top_k_values, top_k_indices = torch.topk(scores, k=k_safe, dim=2, largest=True)

		# Top-K 以外の要素を 0 にした隣接行列 A を作成
		# A は [B, N, N]
		adj_matrix_sparse = torch.zeros_like(scores, device=x.device, dtype=x.dtype)

		# ノードiからノードjへのインデックス [B, N, k_safe]
		row_indices = torch.arange(nodes_per_sample, device=x.device).view(1, -1, 1).expand_as(top_k_indices)
		batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand_as(top_k_indices)

		# Top-K の位置にスコアを scatter (書き込み)
		adj_matrix_sparse[batch_indices, row_indices, top_k_indices] = top_k_values

		# GNN での計算を安定させるため、重みを正規化 (例: Softmax)
		# これが学習可能なエッジ重み (edge_weight) となる
		adj_matrix_normalized = F.softmax(adj_matrix_sparse, dim=2)

		# 4. グラフ正則化損失 (L_reg) の計算
		# L_reg = Tr(X^T L X), L = D - A
		#
		graph_reg_loss = self._calculate_graph_regularization_loss(
			adj_matrix=adj_matrix_normalized,
			features=x_reshaped
		)

		# 5. GNNへの連携 (edge_index, edge_weight の生成)
		# adj_matrix_normalized は [B, N, N] の密行列（中身は疎）
		# PyG の GNNConv は (edge_index, edge_weight) 形式を要求する

		# バッチ処理用にオフセット付きの edge_index を生成
		edge_indices = []
		edge_weights = []
		offset = 0

		for i in range(batch_size):
			# i番目のサンプルの隣接行列 [N, N]
			adj_sample = adj_matrix_normalized[i]

			# 密行列から疎な edge_index と edge_weight を抽出
			edge_index_sample, edge_weight_sample = dense_to_sparse(adj_sample)

			# ノードインデックスにオフセットを加算
			edge_index_sample = edge_index_sample + offset

			edge_indices.append(edge_index_sample)
			edge_weights.append(edge_weight_sample)

			offset += nodes_per_sample

		# バッチ全体で結合
		edge_index = torch.cat(edge_indices, dim=1)
		edge_weight = torch.cat(edge_weights, dim=0)

		return edge_index, edge_weight, graph_reg_loss

	def _calculate_graph_regularization_loss(self,
	                                         adj_matrix: torch.Tensor,
	                                         features: torch.Tensor) -> torch.Tensor:
		"""
		グラフ正則化損失 L_reg = Tr(X^T L X) を計算する


		Args:
			adj_matrix (torch.Tensor): 正規化済み隣接行列 A [B, N, N]
			features (torch.Tensor): ノード特徴量 X [B, N, C]

		Returns:
			torch.Tensor: スカラーの損失値
		"""

		# 1. 次数行列 D の計算
		# D_ii = sum_j(A_ij)
		degree_matrix = torch.sum(adj_matrix, dim=2)  # [B, N]
		# D を対角行列 [B, N, N] に変換
		degree_matrix_diag = torch.diag_embed(degree_matrix)

		# 2. グラフラプラシアン L の計算
		# L = D - A
		#
		laplacian_matrix = degree_matrix_diag - adj_matrix  # [B, N, N]

		# 3. L_reg = Tr(X^T L X) の計算
		# X^T L X は [B, C, C] の行列になる
		# (X^T @ L) @ X
		# [B, C, N] @ [B, N, N] -> [B, C, N]
		xt_l = torch.bmm(features.transpose(1, 2), laplacian_matrix)

		# [B, C, N] @ [B, N, C] -> [B, C, C]
		xt_l_x = torch.bmm(xt_l, features)

		# トレース (対角和) を計算し、バッチ全体で平均
		# torch.diagonal で [B, C] の対角成分を取得し、Cについて合計 -> [B]
		# 最後にバッチで平均
		loss = torch.mean(torch.sum(torch.diagonal(xt_l_x, dim1=-2, dim2=-1), dim=1))

		return loss