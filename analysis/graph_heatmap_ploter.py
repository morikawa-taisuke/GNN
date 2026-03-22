import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class NodeReferenceAnalyzer:
	def __init__(self, input_dir, output_dir):
		self.input_dir = Path(input_dir)
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)

	def visualize_heatmap(self, file_name):
		"""指定されたnpzファイルのノード参照頻度をヒートマップで可視化"""
		path = self.input_dir / file_name
		if not path.exists():
			print(f"Error: {file_name} not found.")
			return

		# 1. データのロード
		data = np.load(path)
		edge_index = data['edge_index']
		coords = data['node_coords']
		num_nodes = coords.shape[0]

		# 2. 各ノードが参照された回数（入次数）をカウント
		# edge_index[1] はターゲット（参照先）ノードのインデックス
		reference_counts = np.bincount(edge_index[1], minlength=num_nodes)

		# 3. 座標系に応じたプロット処理
		if coords.shape[1] == 2:
			# --- 2次元座標の場合 (例: 32 x 13) ---
			h_idx = coords[:, 1].astype(int)
			w_idx = coords[:, 0].astype(int)

			# ヒートマップ用行列の作成
			grid_h = h_idx.max() + 1
			grid_w = w_idx.max() + 1
			heatmap_data = np.zeros((grid_h, grid_w))

			for i in range(num_nodes):
				heatmap_data[h_idx[i], w_idx[i]] = reference_counts[i]

			plt.figure(figsize=(12, 5))
			sns.heatmap(heatmap_data, cmap='magma', annot=False, cbar_kws={'label': 'Reference Count'})
			plt.title(f"Node Reference Heatmap (2D): {file_name}")
			plt.xlabel("Bottleneck Width (Time)")
			plt.ylabel("Bottleneck Height (Freq)")
			plt.gca().invert_yaxis()  # スペクトログラムの向きに合わせる

		else:
			# --- 1次元座標の場合 (GNN.py のデフォルト設定) ---
			# 時間軸に沿った参照頻度をプロット
			time_idx = coords.flatten()

			plt.figure(figsize=(15, 3))
			# 1行のヒートマップとして表示
			heatmap_data = reference_counts.reshape(1, -1)
			sns.heatmap(heatmap_data, cmap='magma', cbar_kws={'label': 'Count'}, yticklabels=False)
			plt.title(f"Temporal Node Reference Density: {file_name}")
			plt.xlabel("Time Index (Bottleneck)")

		# 保存
		save_path = self.output_dir / f"{Path(file_name).stem}_heatmap.png"
		plt.tight_layout()
		plt.savefig(save_path)
		plt.close()
		print(f"✅ Heatmap saved to: {save_path}")


if __name__ == "__main__":
	# --- 設定 ---
	# npzファイルが保存されているディレクトリを指定
	INPUT_DIR = "/Users/a/Documents/sound_data/RESULT/graphs_analysis/Speq_GNN/SpeqGAT_all_knn_noise_only"
	OUTPUT_DIR = "./analysis_heatmaps"

	# 解析したいファイル名
	TARGET_FILE = "p232_001_TCAR_01ch_4db_159deg.npz"

	analyzer = NodeReferenceAnalyzer(INPUT_DIR, OUTPUT_DIR)
	analyzer.visualize_heatmap(TARGET_FILE)