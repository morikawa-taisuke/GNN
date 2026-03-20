import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os


def plot_parallel_model_comparison(input_dir, output_dir="plots"):
	# 1. 保存先ディレクトリの作成
	os.makedirs(output_dir, exist_ok=True)

	# 2. ディレクトリ内のすべてのCSVを取得
	csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
	if not csv_files:
		print(f"Error: No CSV files found in {input_dir}")
		return

	all_dfs = []
	for f in csv_files:
		temp_df = pd.read_csv(f)

		# 【重要】CSVのファイル名を「モデル名」として列に追加（または上書き）
		# 例: "GCN_results.csv" -> "GCN_results"
		model_name = os.path.basename(f).replace('.csv', '')
		temp_df['Model'] = model_name

		all_dfs.append(temp_df)

	# 全データを統合
	df = pd.concat(all_dfs, ignore_index=True)

	# 3. 描画設定
	sns.set_theme(style="whitegrid")
	metrics = ['Kurtosis', 'IslandCount', 'FluxVar']

	# グラフのカラーパレット（モデル数に応じて自動調整）
	palette = sns.color_palette("husl", len(df['Model'].unique()))

	for metric in metrics:
		if metric not in df.columns:
			print(f"Skip: {metric} not found in data.")
			continue

		plt.figure(figsize=(14, 7))

		# 箱ひげ図の描画
		# x軸: 帯域(Band), y軸: 指標値, hue: モデル(ファイル名)で並列化
		ax = sns.boxplot(
			data=df,
			x='Band',
			y=metric,
			hue='Model',
			palette=palette,
			order=['Low', 'Mid', 'High'],
			showfliers=True  # 外れ値（極端なノイズ）も表示
		)

		# 各帯域内でのモデル比較を強調
		plt.title(f'Musical Noise Analysis: {metric} Comparison', fontsize=16, fontweight='bold')
		plt.xlabel('Frequency Band', fontsize=13)
		plt.ylabel(f'Value ({metric})', fontsize=13)

		# 凡例を外側に配置
		plt.legend(title='Models (from CSV names)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

		plt.tight_layout()

		# 4. 保存
		save_path = os.path.join(output_dir, f"compare_{metric}.png")
		plt.savefig(save_path, dpi=300)
		plt.close()
		print(f"Generated comparison plot: {save_path}")


if __name__ == "__main__":
	# 比較したいCSVファイル群が入っているフォルダを指定
	plot_parallel_model_comparison(input_dir="./analysis_results/reverb_only")