import pandas as pd
import os
import glob
import numpy as np


def summarize_graph_analysis(input_dir, output_file="final_graph_summary.csv"):
	"""
	モデルごとのCSVを集計し、比較軸（GNN/Domain/Edge/Range）ごとのサマリー表を作成する
	"""
	csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)

	if not csv_files:
		print(f"Error: No CSV files found in {input_dir}")
		return

	all_data_list = []

	for f in csv_files:
		# 空ファイルのチェック (EmptyDataError対策)
		if os.path.getsize(f) == 0:
			print(f"Skip empty file: {f}")
			continue

		try:
			# 1. 個別CSVの読み込み
			df = pd.read_csv(f)
			if df.empty:
				continue

			# 2. ファイル名から比較軸（メタデータ）を抽出
			# 命名規則: GNN_Domain_Edge_Range.csv を想定
			# 例: SpeqGAT_temporal_random_reverb_only.csv
			base_name = os.path.basename(f).replace(".csv", "")
			parts = base_name.split("_")

			# 要素が足りない場合のデフォルト値設定
			gnn_type = parts[0] if len(parts) > 0 else "Unknown"
			domain = parts[0] if len(parts) > 1 else "Unknown"
			edge_met = parts[2] if len(parts) > 2 else "Unknown"
			e_range = parts[1] if len(parts) > 3 else "Unknown"
			wave_type = f"{parts[3]}_{parts[4]}" if len(parts) > 4 else "Unknown"

			# メタデータ列を追加
			df['GNN'] = gnn_type
			df['Domain'] = domain
			df['Edge'] = edge_met
			df['Range'] = e_range
			df['WaveType'] = wave_type

			all_data_list.append(df)

		except Exception as e:
			print(f"Error processing {f}: {e}")

	if not all_data_list:
		print("No valid data to summarize.")
		return

	# 3. 全データの結合
	combined_df = pd.concat(all_data_list, ignore_index=True)

	# 集計対象の数値列を取得
	numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
	# FileNameなどは集計から除外
	if 'FileName' in numeric_cols: numeric_cols.remove('FileName')

	# 4. モデル構成ごとに統計量（平均・標準偏差・最大値）を計算
	# 2以降の内容: 比較軸でグループ化して統計量を算出
	summary_mean = combined_df.groupby(['GNN', 'Domain', 'Edge', 'Range', 'WaveType'])[numeric_cols].mean().add_suffix('_mean')
	summary_std = combined_df.groupby(['GNN', 'Domain', 'Edge', 'Range', 'WaveType'])[numeric_cols].std().add_suffix('_std')
	summary_max = combined_df.groupby(['GNN', 'Domain', 'Edge', 'Range', 'WaveType'])[numeric_cols].max().add_suffix('_max')

	# 全ての統計量を横に連結
	final_summary = pd.concat([summary_mean, summary_std, summary_max], axis=1).reset_index()

	# 5. 結果の保存
	final_summary.to_csv(output_file, index=False)
	print(f"\n✅ Summary table created: {output_file}")

	# プレビュー表示
	print("\n--- Summary Preview (Means) ---")
	print(final_summary[['GNN', 'Domain', 'Edge', 'Range', 'WaveType'] + [c for c in final_summary.columns if '_mean' in c]].head())

	return final_summary


if __name__ == "__main__":
	# 個別解析済みCSVが格納されているディレクトリを指定
	# 例: /Users/a/Documents/sound_data/RESULT/graphs_analysis
	target_dir = "/Users/a/Documents/sound_data/RESULT/graphs_analysis"
	summarize_graph_analysis(input_dir=target_dir)