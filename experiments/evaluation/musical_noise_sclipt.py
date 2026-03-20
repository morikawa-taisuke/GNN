"""
ミュージカルノイズの発生に関するcsvをまとめるためのスクリプト

機能:
    - summarize_musical_noise_from_csv: CSVファイルを読み込み、メタデータを統合し、
      要約統計量を計算し、集計された要約を含むCSVファイルを出力します。
"""
import pandas as pd
import os
import glob
import numpy as np


def summarize_musical_noise_from_csv(input_dir, output_file="musical_noise_final_summary.csv"):
	"""
	CSV内の'Model'列と、ファイル名に含まれる条件（Domain/Edge/Range）を統合して集計する
	"""
	csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)

	if not csv_files:
		print(f"Error: No CSV files found in {input_dir}")
		return

	all_data_list = []

	for f in csv_files:
		if os.path.getsize(f) == 0: continue

		try:
			df = pd.read_csv(f)
			if df.empty: continue

			# --- ファイル名から実験条件を抽出 ---
			# 命名規則例: WaveGNN_noise_only_all_knn.csv
			base_name = os.path.basename(f).replace(".csv", "")
			parts = base_name.split("_")

			domain_attr = parts[0] if len(parts) > 0 else "Unknown"
			range_attr = parts[3] if len(parts) > 2 else "Unknown"
			edge_attr = parts[4] if len(parts) > 3 else "Unknown"
			wave_type = f"{parts[1]}_{parts[2]}" if len(parts) > 1 else "Unknown"

			# 既存の列にファイル由来のメタデータを追加
			df['Domain'] = domain_attr
			df['Range'] = range_attr
			df['Edge'] = edge_attr
			df['WaveType'] = wave_type

			all_data_list.append(df)

		except Exception as e:
			print(f"Error processing {f}: {e}")

	if not all_data_list:
		print("No valid data found.")
		return

	# 1. 全データの結合
	combined_df = pd.concat(all_data_list, ignore_index=True)

	# 2. 数値指標の特定
	metrics = ['Kurtosis', 'IslandCount', 'FluxVar']
	available_metrics = [m for m in metrics if m in combined_df.columns]

	# 3. 集計のキーとなる列を指定
	# CSV内部の 'Model', 'Band' と、ファイル由来のメタデータをすべて使う
	group_cols = ['Model', 'Domain', 'Edge', 'Range', 'Band', 'WaveType']

	# 4. 統計量（平均・標準偏差・最大値）を計算
	summary_mean = combined_df.groupby(group_cols)[available_metrics].mean().add_suffix('_mean')
	summary_std = combined_df.groupby(group_cols)[available_metrics].std().add_suffix('_std')
	summary_max = combined_df.groupby(group_cols)[available_metrics].max().add_suffix('_max')

	# 結果を横に連結
	final_summary = pd.concat([summary_mean, summary_std, summary_max], axis=1).reset_index()
	print(final_summary[['WaveType'] + [c for c in final_summary.columns if '_mean' in c]].head())
	# print(final_summary.head(10))
	# exit()

	# 5. 保存
	final_summary.to_csv(output_file, index=False)
	print(f"\n✅ Summary created: {output_file}")

	# 表示用に少し整形してプレビュー
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', 1000)
	print("\n--- Summary Preview ---")
	print(final_summary.head(10))

	return final_summary


if __name__ == "__main__":
	# 解析済みCSVが格納されているフォルダを指定
	target_dir = "/Users/a/Documents/sound_data/RESULT/musical_noise/"
	summarize_musical_noise_from_csv(input_dir=target_dir)
