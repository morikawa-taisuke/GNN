import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os


def plot_graph_comparison(input_dir, output_dir="plots_graph", wave_type="noise_only"):
	os.makedirs(output_dir, exist_ok=True)

	# 1. すべてのグラフ解析CSVを読み込む
	csv_files = glob.glob(os.path.join(input_dir, "**", f"*{wave_type}.csv"), recursive=True)
	if not csv_files:
		print(f"Error: No CSV files found in {input_dir}")
		return

	print(csv_files)

	all_dfs = []
	for f in csv_files:
		temp_df = pd.read_csv(f)
		# ファイル名をモデル名として使用
		model_name = os.path.basename(f).replace('.csv', '')
		temp_df['Model'] = model_name
		all_dfs.append(temp_df)

	df = pd.concat(all_dfs, ignore_index=True)

	# 2. 比較したいグラフ指標のリスト
	# (前回定義した指標: Avg_Freq_Dist, Dirichlet_Energy, attention_layer_0_entropy など)
	metrics = [c for c in df.columns if c not in ['FileName', 'Model', 'Band']]

	sns.set_theme(style="whitegrid")

	for metric in metrics:
		plt.figure(figsize=(12, 6))

		# モデルごとの分布を比較
		ax = sns.boxplot(
			data=df,
			x='Model',
			y=metric,
			palette="husl"
		)

		plt.title(f'Graph Structural Analysis: {metric}', fontsize=15, fontweight='bold')
		plt.ylabel(f'Value: {metric}', fontsize=12)
		plt.xlabel('Model Configuration', fontsize=12)
		plt.xticks(rotation=45)  # モデル名が長い場合に備えて傾ける

		plt.tight_layout()
		save_path = os.path.join(output_dir, f"graph_compare_{metric}_{wave_type}.png")
		plt.savefig(save_path, dpi=300)
		plt.close()
		print(f"Generated plot: {save_path}")


if __name__ == "__main__":
	# グラフ解析結果のCSVが保存されているディレクトリを指定
	wave_type_list = ["noise_only", "reverb_only", "noise_reverb"]
	for wave_type in wave_type_list:
		plot_graph_comparison(
			input_dir="/Users/a/Documents/sound_data/RESULT/graphs_analysis",
			output_dir=f"/Users/a/Documents/sound_data/RESULT/graphs_analysis/plots_{wave_type}",
			wave_type=wave_type
		)