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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_attention_heatmap(npz_path, layer_idx=0, head_idx=0):
	# 1. データの読み込み
	data = np.load(npz_path)
	edge_index = data['edge_index']  # [2, E]

	# ノード数を取得 (最大のインデックス + 1)
	num_nodes = edge_index.max() + 1

	# アテンション重みの取得
	att_key = f'attention_layer_{layer_idx}'
	if att_key in data.files:
		weights = data[att_key]  # [E, heads] または [E, 1]
		if weights.ndim > 1:
			# 指定したヘッドの重みを使用
			weights = weights[:, head_idx]
	else:
		# 重みがない場合は一律1にする (GCNの場合など)
		weights = np.ones(edge_index.shape[1])

	# 2. 隣接行列（ヒートマップ用）の構築
	# 行：出発ノード (Source), 列：到着ノード (Target)
	adj_matrix = np.zeros((num_nodes, num_nodes))
	adj_matrix[edge_index[0], edge_index[1]] = weights

	# 3. プロット
	plt.figure(figsize=(10, 8))
	sns.heatmap(adj_matrix, cmap='viridis', cbar_kws={'label': 'Attention Weight'})

	plt.title(f"Attention Heatmap (Layer {layer_idx}, Head {head_idx})\n{np.path.basename(npz_path)}")
	plt.xlabel("Target Node (Arrival)")
	plt.ylabel("Source Node (Departure)")

	# 軸のメモリが多いと見づらいため、適度に間引く
	step = max(1, num_nodes // 10)
	plt.xticks(np.arange(0, num_nodes, step), np.arange(0, num_nodes, step))
	plt.yticks(np.arange(0, num_nodes, step), np.arange(0, num_nodes, step))

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	# 実行
	plot_attention_heatmap("/Users/a/Documents/sound_data/RESULT/graphs_analysis/Speq_GNN/SpeqGAT_all_knn_noise_only/p232_001_TCAR_01ch_4db_159deg.npz")
	# visualize_graph_edges("/Users/a/Documents/sound_data/RESULT/graphs_analysis/Speq_GNN/SpeqGAT_all_knn_noise_only/p232_001_TCAR_01ch_4db_159deg.npz")
