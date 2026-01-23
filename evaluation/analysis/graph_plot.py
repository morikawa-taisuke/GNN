"""
NPZファイルに保存されたグラフの辺を可視化します。

この関数は、NPZファイルで表現されたグラフの辺を可視化します。
グラフは、2次元座標（周波数と時間）を持つノードと、オプションで注意値で重み付けされた辺で構成されます。
辺の総数が指定された上限を超える場合、レンダリング性能を最適化するために辺のサブサンプルを可視化できます。

:param npz_path: グラフデータを含むNPZファイルへのパス。
:type npz_path: str
:param layer_idx: 重みを抽出するアテンション層のインデックス。デフォルトは0。
:type layer_idx: int, optional
:param max_edges: 表示するエッジの最大数。総数がこの値を超える場合、エッジはランダムにサンプリングされる。
    デフォルトは2000。
:type max_edges: int, optional

:data edge_index: 各エッジの送信元と送信先インデックス、形状 [2, E]、
    NPZファイルから抽出。
    (E: エッジ数)。
:type edge_index: numpy.ndarray
:data node_coords: ノードの2次元座標（周波数と時間形式）、
    形状 [N, 2] (N: ノード数)。
:type node_coords: numpy.ndarray
:data weights: エッジに対する注意重み。NPZファイル内の注意層から取得。
    形状は注意設定により異なる。オプション。
:type weights: numpy.ndarray, optional
:raises FileNotFoundError: 指定されたNPZファイルが見つからない場合。
:raises ValueError: NPZファイルに必要なキーが存在しない場合。

:return: この関数には戻り値はありませんが、ノードとサンプリングされたエッジを可視化したmatplotlibプロットを表示します。
:rtype: None
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os


def visualize_graph_edges(npz_path, layer_idx=0, max_edges=2000):
	# 1. データの読み込み
	data = np.load(npz_path)
	edge_index = data['edge_index']  # [2, E]
	node_coords = data['node_coords']  # [N, 2] (Freq, Time)

	# アテンション重みがある場合 (GAT)
	att_key = f'attention_layer_{layer_idx}'
	weights = data[att_key] if att_key in data.files else None
	if weights is not None and weights.ndim > 1:
		weights = weights.mean(axis=1)  # マルチヘッドの平均をとる

	# 2. 描画データの準備
	# 全て描画すると重いため、ランダムにサンプリング
	num_edges = edge_index.shape[1]
	if num_edges > max_edges:
		indices = np.random.choice(num_edges, max_edges, replace=False)
		edge_index = edge_index[:, indices]
		if weights is not None:
			weights = weights[indices]

	src_idx = edge_index[0]
	dst_idx = edge_index[1]

	# 座標の取得 (Timeがx軸、Freqがy軸になるように入れ替え)
	# SpeqGNNのcoordsは [Freq_idx, Time_idx] と想定
	x_coords = node_coords[:, 1]
	y_coords = node_coords[:, 0]

	segments = []
	for s, d in zip(src_idx, dst_idx):
		segments.append([(x_coords[s], y_coords[s]), (x_coords[d], y_coords[d])])

	# 3. プロット
	fig, ax = plt.subplots(figsize=(12, 8))

	# ノードを点として描画
	ax.scatter(x_coords, y_coords, s=5, c='blue', alpha=0.3, label='Nodes (T-F bins)')

	# エッジを線として描画
	if weights is not None:
		# 重みに基づいて色の濃さを変える
		lc = LineCollection(segments, array=weights, cmap='viridis', alpha=0.5, linewidths=0.5)
		ax.add_collection(lc)

		# ax=ax を追加して、カラーバーを配置する軸を指定する
		fig.colorbar(lc, ax=ax, label='Attention Weight')
	else:
		lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.5)

	ax.add_collection(lc)

	ax.set_title(f"Graph Edge Visualization: {os.path.basename(npz_path)}")
	ax.set_xlabel("Time Frame (Bottleneck)")
	ax.set_ylabel("Frequency Bin (Bottleneck)")
	ax.legend()
	plt.grid(True, alpha=0.2)
	plt.show()

if __name__ == "__main__":
	visualize_graph_edges("/Users/a/Documents/sound_data/RESULT/graphs_analysis/Speq_GNN/SpeqGAT_all_knn_noise_only/p232_001_TCAR_01ch_4db_159deg.npz")
