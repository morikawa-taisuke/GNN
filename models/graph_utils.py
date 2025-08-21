import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch


class NodeSelectionType(Enum):
    ALL = "all"  # 全ノードから選択
    TEMPORAL = "temporal"  # 時間窓内のノードから選択


class EdgeSelectionType(Enum):
    RANDOM = "random"  # ランダムに選択
    KNN = "knn"  # 類似度に基づいて選択


@dataclass
class GraphConfig:
    """グラフ作成の設定を保持するデータクラス"""

    num_edges: int  # 各ノードから張るエッジの数
    node_selection: NodeSelectionType  # ノード選択の方法
    edge_selection: EdgeSelectionType  # エッジ選択の方法
    temporal_window: Optional[int] = 4000  # 時間窓のサイズ（TEMPORALの場合）
    use_self_loops: bool = False  # 自分自身へのループを張るかどうか
    bidirectional: bool = True  # 双方向エッジを張るかどうか


class GraphBuilder:
    def __init__(self, config: GraphConfig):
        self.config = config

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

        # 現在のノードと候補ノードの特徴量を取得
        current_feat = features[node_idx].unsqueeze(0)  # [1, D]
        candidate_feats = features[candidates]  # [N, D]

        # コサイン類似度を計算
        similarities = torch.nn.functional.cosine_similarity(current_feat, candidate_feats)

        # 最も類似度の高いk個のノードを選択
        k = min(num_select, len(candidates))
        _, top_indices = similarities.topk(k)

        return [candidates[idx] for idx in top_indices.tolist()]

    def create_graph(self, num_nodes: int, device: torch.device, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """グラフを作成"""
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        # RANDOM は既存ロジック（ノード毎のサンプリング）を維持
        if self.config.edge_selection == EdgeSelectionType.RANDOM:
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

        # ここから KNN の高速ベクトル化実装
        if features is None:
            raise ValueError("KNNにはノードの特徴量が必要です")

        with torch.no_grad():
            x = features.to(device)  # [N, D]
            N = x.size(0)
            # cos 類似度用に L2 正規化
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            sim = x_norm @ x_norm.T  # [N, N]

            # 候補マスクを作成（True が有効候補）
            if self.config.node_selection == NodeSelectionType.ALL:
                valid = torch.ones((N, N), dtype=torch.bool, device=device)
            else:
                w = int(self.config.temporal_window or self.config.num_edges)
                idx = torch.arange(N, device=device)
                # |i - j| <= w を許可
                ii = idx.view(-1, 1)
                jj = idx.view(1, -1)
                valid = (jj - ii).abs() <= w

            # 自己ループの扱い
            if not self.config.use_self_loops:
                valid.fill_diagonal_(False)

            # 無効候補を -inf にマスク
            neg_inf = torch.finfo(sim.dtype).min
            sim_masked = sim.masked_fill(~valid, neg_inf)

            # topk の k は上限として「利用可能候補数」を超えない値に設定
            # ALL なら N-1（自己ループ無効の場合）/ N（有効の場合）
            # TEMPORAL は最小候補数に不足が出るが、後で -inf を除外して補正する
            max_possible = valid.sum(dim=1).min().item()
            k = min(self.config.num_edges, max(1, int(max_possible)))

            # 近傍抽出
            top_vals, top_idx = torch.topk(sim_masked, k=k, dim=1, largest=True, sorted=False)  # [N, k] each

            # -inf（無効）を除外
            valid_sel = torch.isfinite(top_vals)  # [N, k]
            rows = torch.arange(N, device=device).unsqueeze(1).expand_as(top_idx)  # [N, k]
            src = rows[valid_sel].reshape(-1)
            dst = top_idx[valid_sel].reshape(-1)

            if src.numel() == 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)

            if self.config.bidirectional:
                # 片方向KNNに対して双方向エッジを追加（重複は問題にならないが unique 化も可能）
                src_bi = torch.cat([src, dst], dim=0)
                dst_bi = torch.cat([dst, src], dim=0)
                edge_index = torch.stack([src_bi, dst_bi], dim=0)
            else:
                edge_index = torch.stack([src, dst], dim=0)

            return edge_index

    def create_batch_graph(
        self, x: torch.Tensor, batch_size: int, nodes_per_sample: int, return_batch_indices: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """バッチ処理用のグラフを作成"""
        if nodes_per_sample == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            batch_indices = torch.empty(0, dtype=torch.long, device=x.device)
        else:
            batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(nodes_per_sample)

            edge_indices = []
            offset = 0

            # 可能な限り autograd 無効化（グラフ構築は微分不要）
            with torch.no_grad():
                for i in range(batch_size):
                    # バッチ内の特徴量を抽出
                    batch_features = x[i * nodes_per_sample : (i + 1) * nodes_per_sample]

                    # バッチごとのグラフを生成（KNNはベクトル化経路）
                    batch_edge_index = self.create_graph(num_nodes=nodes_per_sample, device=x.device, features=batch_features)

                    # ノードインデックスをオフセット
                    batch_edge_index = batch_edge_index + offset
                    edge_indices.append(batch_edge_index)
                    offset += nodes_per_sample

            edge_index = torch.cat(edge_indices, dim=1)

        if return_batch_indices:
            return edge_index, batch_indices
        return edge_index