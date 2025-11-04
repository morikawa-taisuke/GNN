# models/graph_utils.py

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
# PyTorch Geometric の grid 関数をインポート
from torch_geometric.utils import grid


class NodeSelectionType(Enum):
	ALL = "all"  # 全ノードから選択
	TEMPORAL = "temporal"  # 時間窓内のノードから選択


class EdgeSelectionType(Enum):
	RANDOM = "random"  # ランダムに選択
	KNN = "knn"  # 類似度に基づいて選択
	GRID = "grid"  # 2D格子グラフ (★追加)


@dataclass
class GraphConfig:
	"""グラフ作成の設定を保持するデータクラス"""

	num_edges: int  # 各ノードから張るエッジの数 (GRIDの場合は無視される)
	node_selection: NodeSelectionType  # ノード選択の方法 (GRIDの場合は無視される)
	edge_selection: EdgeSelectionType  # エッジ選択の方法
	temporal_window: Optional[int] = 4000  # 時間窓のサイズ（TEMPORALの場合）
	use_self_loops: bool = False  # 自分自身へのループを張るかどうか
	bidirectional: bool = True  # 双方向エッジを張るかどうか


class GraphBuilder:
	def __init__(self, config: GraphConfig):
		self.config = config

	# ... (既存の _get_candidate_nodes, _select_edges_random, _select_edges_knn は変更なし) ...
	# ( ... _get_candidate_nodes, _select_edges_random, _select_edges_knn のコード ... )

	def _get_candidate_nodes(self, node_idx: int, num_nodes: int) -> torch.Tensor:
		"""ノードの候補を取得"""
		if self.config.node_selection == NodeSelectionType.ALL:
			# 全ノードから自分自身を除いた候補を返す
			candidates = list(range(0, node_idx)) + list(range(node_idx + 1, num_nodes))
		else:  # TEMPORAL
			window = self.config.temporal_window or self.config.num_edges
			start_idx = max(0, node_idx - window)
			end_idx = min(num_nodes, node_idx + window + 1)
			candidates = list(range(start_idx, node_idx)) + list(range(node_idx + 1, end_idx))
		return candidates

	def _select_edges_random(
			self, node_idx: int, candidates: list, num_select: int, features: Optional[torch.Tensor] = None
	) -> list:
		"""ランダムにエッジを選択"""
		return random.sample(candidates, min(num_select, len(candidates)))

	def _select_edges_knn(self, node_idx: int, candidates: list, num_select: int, features: torch.Tensor) -> list:
		"""類似度に基づいてエッジを選択"""
		if len(candidates) == 0:
			return []
		current_feat = features[node_idx].unsqueeze(0)
		candidate_feats = features[candidates]
		similarities = torch.nn.functional.cosine_similarity(current_feat, candidate_feats)
		k = min(num_select, len(candidates))
		_, top_indices = similarities.topk(k)
		return [candidates[idx] for idx in top_indices.tolist()]

	def create_graph(self, num_nodes: int, device: torch.device, features: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""グラフを作成 (★この関数のロジックは create_batch_graph に移行)"""

		# ★★★ ロジックを create_batch_graph に集約するため、
		#     KNNとRANDOMのロジックをこちらに移動（または維持）します。
		#     （前回の実装では create_batch_graph が create_graph を呼んでいました）

		if num_nodes <= 1:
			return torch.empty((2, 0), dtype=torch.long, device=device)

		if self.config.edge_selection == EdgeSelectionType.GRID:
			raise NotImplementedError("GRID graph creation requires height and width, use create_batch_graph.")

		if self.config.edge_selection == EdgeSelectionType.RANDOM:
			# ( ... 既存のRANDOMロジック ... )
			source_nodes = []
			target_nodes = []
			for i in range(num_nodes):
				candidates = self._get_candidate_nodes(i, num_nodes)
				selected_nodes = self._select_edges_random(i, candidates, self.config.num_edges, features)
				source_nodes.extend([i] * len(selected_nodes))
				target_nodes.extend(selected_nodes)
				if self.config.bidirectional:
					source_nodes.extend(selected_nodes)
					target_nodes.extend([i] * len(selected_nodes))
			edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long, device=device)
			return edge_index

		if features is None:
			raise ValueError("KNNにはノードの特徴量が必要です")

		with torch.no_grad():
			# ( ... 既存のKNNロジック ... )
			x = features.to(device)
			N = x.size(0)
			x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
			sim = x_norm @ x_norm.T
			if self.config.node_selection == NodeSelectionType.ALL:
				valid = torch.ones((N, N), dtype=torch.bool, device=device)
			else:
				w = int(self.config.temporal_window or self.config.num_edges)
				idx = torch.arange(N, device=device)
				ii = idx.view(-1, 1)
				jj = idx.view(1, -1)
				valid = (jj - ii).abs() <= w
			if not self.config.use_self_loops:
				valid.fill_diagonal_(False)
			neg_inf = torch.finfo(sim.dtype).min
			sim_masked = sim.masked_fill(~valid, neg_inf)
			max_possible = valid.sum(dim=1).min().item()
			k = min(self.config.num_edges, max(1, int(max_possible)))
			top_vals, top_idx = torch.topk(sim_masked, k=k, dim=1, largest=True, sorted=False)
			valid_sel = torch.isfinite(top_vals)
			rows = torch.arange(N, device=device).unsqueeze(1).expand_as(top_idx)
			src = rows[valid_sel].reshape(-1)
			dst = top_idx[valid_sel].reshape(-1)
			if src.numel() == 0:
				return torch.empty((2, 0), dtype=torch.long, device=device)
			if self.config.bidirectional:
				src_bi = torch.cat([src, dst], dim=0)
				dst_bi = torch.cat([dst, src], dim=0)
				edge_index = torch.stack([src_bi, dst_bi], dim=0)
			else:
				edge_index = torch.stack([src, dst], dim=0)
			return edge_index

	def create_batch_graph(
			self,
			x_features_4d: torch.Tensor,  # ★ 入力を4Dテンソルに変更
			return_batch_indices: bool = False
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		"""
        バッチ処理用のグラフを作成。
        入力 x_features_4d は [Batch, Channels, Height, Width] の形状を仮定。
        """

		# ★ 1. 4Dテンソルから形状情報を推論
		if x_features_4d.dim() != 4:
			raise ValueError(
				f"create_batch_graph は 4D テンソル (B, C, H, W) を想定していますが、{x_features_4d.dim()}D が入力されました。")

		batch_size, channels, height, width = x_features_4d.size()
		nodes_per_sample = height * width
		device = x_features_4d.device

		batch_indices = None
		if nodes_per_sample == 0:
			edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
			if return_batch_indices:
				batch_indices = torch.empty(0, dtype=torch.long, device=device)

		# ★ 2. GRID グラフ用のロジック
		elif self.config.edge_selection == EdgeSelectionType.GRID:
			# 1. 単一の2D格子グラフを作成 (4近傍、自己ループなし)
			#    torch_geometric.utils.grid は自動で双方向エッジを生成します
			single_edge_index = grid(height=height, width=width, device=device)[0]

			# 2. バッチ数分だけオフセットを加えて連結
			edge_indices = []
			offset = 0
			for i in range(batch_size):
				edge_indices.append(single_edge_index + offset)
				offset += nodes_per_sample  # (height * width)

			edge_index = torch.cat(edge_indices, dim=1)

			if return_batch_indices:
				batch_indices = torch.arange(batch_size, device=device).repeat_interleave(nodes_per_sample)

		# ★ 3. 既存の RANDOM / KNN のロジック
		else:
			if return_batch_indices:
				batch_indices = torch.arange(batch_size, device=device).repeat_interleave(nodes_per_sample)

			# 特徴量を KNN/RANDOM が期待するフラットな形式 [B*N, C] に変形
			# (SpeqGNN.py 側でも GNN のために同じ変形を行いますが、
			#  GraphBuilder の責務としてここでも変形を行います)
			x_nodes_flat = x_features_4d.view(batch_size, channels, -1).permute(0, 2, 1).reshape(-1, channels)

			edge_indices = []
			offset = 0

			with torch.no_grad():
				for i in range(batch_size):
					# バッチ内の特徴量を抽出 [N, C]
					batch_features = x_nodes_flat[i * nodes_per_sample: (i + 1) * nodes_per_sample]

					# バッチごとのグラフを生成
					batch_edge_index = self.create_graph(num_nodes=nodes_per_sample, device=device, features=batch_features)

					# ノードインデックスをオフセット
					batch_edge_index = batch_edge_index + offset
					edge_indices.append(batch_edge_index)
					offset += nodes_per_sample

			if len(edge_indices) > 0:
				edge_index = torch.cat(edge_indices, dim=1)
			else:
				edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

		if return_batch_indices:
			return edge_index, batch_indices
		return edge_index