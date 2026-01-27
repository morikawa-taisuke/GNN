"""
グラフ構造に関して分析するプログラム．

このモジュールは、.npzファイルに保存されたエクスポート済みグラフデータを分析する機能を提供します。
分析対象には距離メトリクス、ノード特徴量の平滑性、アテンション分析が含まれます。
分析結果は指定されたCSVファイルに保存されます。

関数:
    analyze_exported_graphs: グラフデータファイルを処理し、メトリクスを計算して結果をCSVファイルに保存します。
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType
from mymodule import my_func, const, LossFunction, confirmation_GPU


def analyze_exported_graphs(input_dir, output_csv="graph_analysis_results.csv"):
	npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
	all_results = []

	for file_path in tqdm(npz_files):
		filename = os.path.basename(file_path)
		data = np.load(file_path)

		# データの取得
		edge_index = data['edge_index']  # [2, E]
		node_coords = data['node_coords']  # [N, 1] (Time) or [N, 2] (Freq, Time)
		node_features = data['node_features']  # [N, D]

		# 1. エッジの物理的距離の解析
		src, dst = edge_index[0], edge_index[1]
		coord_src = node_coords[src]
		coord_dst = node_coords[dst]

		# 距離計算 (時間軸・周波数軸)
		dist_diff = np.abs(coord_src - coord_dst)
		if node_coords.shape[1] == 2:  # 周波数ドメイン (SpeqGNN)
			f_dist = dist_diff[:, 0]
			t_dist = dist_diff[:, 1]
		else:  # 時間ドメイン (UGNN)
			f_dist = np.zeros_like(dist_diff[:, 0])
			t_dist = dist_diff[:, 0]

		# 2. 特徴量の平滑度 (Graph Dirichlet Energy)
		# 隣接ノード間の特徴量の差の二乗和
		feat_diff = np.sum((node_features[src] - node_features[dst]) ** 2, axis=1)
		dirichlet_energy = np.mean(feat_diff)

		# 3. アテンションの解析 (GATの場合)
		attention_metrics = {}
		for key in data.keys():
			if 'attention_layer' in key:
				att_weights = data[key]  # [E, heads] または [E, 1]
				# 各ノード(dst)に入ってくる重みの合計を1にするため、ノード単位で集計
				# ここでは簡易的に全エッジの重みのエントロピーの平均を計算
				# 本来はdstごとにsoftmax後の分布を見るのが理想的
				avg_entropy = np.mean([entropy(att_weights[:, h]) for h in range(att_weights.shape[1])])
				attention_metrics[f'{key}_entropy'] = avg_entropy

		# 結果をまとめる
		res = {
			'FileName': filename,
			'Avg_Freq_Dist': np.mean(f_dist),
			'Max_Freq_Dist': np.max(f_dist),
			'Avg_Time_Dist': np.mean(t_dist),
			'Dirichlet_Energy': dirichlet_energy,
		}
		res.update(attention_metrics)
		all_results.append(res)

	# 結果をCSVに保存
	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	df = pd.DataFrame(all_results)
	df.to_csv(output_csv, index=False)
	print(f"\nSaved graph analysis to {output_csv}")
	return df


if __name__ == "__main__":
	# 実行例
	model_list = [
		"UGCN",
		# "UGAT",
		# "SpeqGCN",
		# "SpeqGAT",
	]  # モデルの種類  "UGCN", "UGAT", "ConvTasNet", "UNet"
	wave_types = [
		"noise_only",
		"reverb_only",
		"noise_reverb",
	]  # 入力信号の種類 (noise_only, reverb_only, noise_reverb)    # UGAT_all_random_reverb_only
	node_selection_list = [
		NodeSelectionType.TEMPORAL,
		# NodeSelectionType.ALL
	]  # ノード選択の方法 (ALL, TEMPORAL)
	edge_selection_list = [
		EdgeSelectionType.RANDOM,
		EdgeSelectionType.KNN
	]  # エッジ選択の方法 (RANDOM, KNN)
	for model_type in model_list:
		for nose_selection in node_selection_list:
			for edge_selection in edge_selection_list:
				for wave_type in wave_types:
					target_dir = f"{const.RESULT_DIR}/graphs_analysis/Wave_GNN/{model_type}_{nose_selection.value}_{edge_selection.value}_{wave_type}"
					output_path = f"{const.RESULT_DIR}/graphs_analysis/result/Wave_GNN/{model_type}/{model_type}_{nose_selection.value}_{edge_selection.value}_{wave_type}.csv"
					df = analyze_exported_graphs(input_dir=target_dir, output_csv=output_path)
