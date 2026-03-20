import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_gnn_plots(csv_path, domain="時間領域", task="雑音抑圧"):
	"""
	指定されたドメインとタスクに基づいて箱ヒゲ図を作成しPDFで保存する
	domain: "時間領域" または "周波数領域"
	task: "雑音抑圧", "残響除去", "同時処理"
	"""
	# 1. データの読み込みとヘッダーの整理
	df_raw = pd.read_csv(csv_path, header=None)

	# 0-3行目のヘッダー情報を結合して列名を作成
	header = df_raw.iloc[:4].ffill(axis=1)
	cols = []
	for i in range(len(df_raw.columns)):
		if i < 3:
			cols.append(f"meta_{i}")
		else:
			# ドメイン_タスク_指標_統計量 (例: 時間領域_雑音抑圧_PESQ_AVE)
			cols.append(f"{header.iloc[0, i]}_{header.iloc[1, i]}_{header.iloc[2, i]}_{header.iloc[3, i]}")

	data = df_raw.iloc[4:].copy()
	data.columns = cols

	# モデル名の整理 (Unnamed列を統合)
	data['Model'] = data['meta_0'].fillna(data['meta_1'])

	# 数値データに変換
	for col in data.columns:
		if "AVE" in col:
			data[col] = pd.to_numeric(data[col], errors='coerce')

	metrics = ['PESQ', 'STOI', 'SI-SDR']
	generated_files = []

	# 2. 各指標ごとにグラフを作成
	for metric in metrics:
		col_name = f"{domain}_{task}_{metric}_AVE"
		if col_name not in data.columns:
			print(f"Warning: {col_name} not found in data.")
			continue

		plt.figure(figsize=(10, 6))

		# 箱ヒゲ図の作成
		# データポイントを重ねて表示することで、試行ごとのばらつきを確認しやすくします
		plot_df = data[['Model', col_name]].dropna()
		sns.boxplot(x='Model', y=col_name, data=plot_df, palette="Set3")
		sns.stripplot(x='Model', y=col_name, data=plot_df, color='black', alpha=0.5)

		plt.title(f"Comparison: {domain} - {task} ({metric})", fontsize=14)
		plt.xlabel("Model / Architecture", fontsize=12)
		plt.ylabel(f"{metric} (Average Value)", fontsize=12)
		plt.xticks(rotation=45)
		plt.grid(axis='y', linestyle='--', alpha=0.7)
		plt.tight_layout()

		# PDFとして保存
		filename = f"{domain}_{task}_{metric}.pdf"
		plt.savefig(filename)
		generated_files.append(filename)
		plt.close()
		print(f"Saved: {filename}")

	return generated_files


# 実行例
if __name__ == "__main__":
	# ファイルパス、ドメイン、タスクを指定して実行
	files = generate_gnn_plots(
		'/Users/a/Documents/sound_data/RESULT/evaluation/Random_Dataset_VCTK_DEMAND_1ch/GNN_data.csv',
		domain="時間領域",
		task="雑音抑圧"
	)