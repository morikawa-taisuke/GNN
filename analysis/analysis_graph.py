# analyze_hdf5_correlation_noisy_graph.py
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from collections import Counter
from tqdm import tqdm
import argparse
import math
import sys
from pathlib import Path  # Pathオブジェクトを使用するためにインポート
from mymodule import const

def load_data_from_hdf5(h5_file, file_key):
	"""指定されたキー (ファイル名) のデータをHDF5ファイルから読み込む"""
	if file_key not in h5_file:
		print(f"⚠️ キー '{file_key}' がファイル内に存在しません。スキップします。")
		return None

	try:
		group = h5_file[file_key]
		# 必要なデータセットが存在するか確認 (clean_index は不要に)
		required_datasets = ['noisy_node', 'clean_node', 'error_node', 'noisy_index']
		if not all(dset in group for dset in required_datasets):
			print(f"⚠️ キー '{file_key}' 内に必要なデータセットが不足しています。スキップします。")
			return None

		data = {
			'noisy_node': group['noisy_node'][:],
			'clean_node': group['clean_node'][:],
			'error_node': group['error_node'][:],
			'noisy_index': group['noisy_index'][:],
			# 'clean_index': group['clean_index'][:] # clean_index は読み込まない
		}
		# エッジインデックスの形状を確認 (空でないことを確認)
		if data['noisy_index'].shape[0] == 0:
			# print(f"情報: キー '{file_key}' には noisy_index データがありません。")
			pass  # エッジがなくても誤差集計は可能

		return data
	except Exception as e:
		print(f"⚠️ キー '{file_key}' のデータ読み込み中にエラーが発生しました: {e}。スキップします。")
		return None


def analyze_hdf5_correlation_noisy_graph(hdf5_file_path, n_fft, hop_length, win_length, num_downsampling_layers, output_dir="."):
	"""
	HDF5ファイル全体を解析し、ノード誤差と**ノイズありグラフの**入次数の相関を計算・可視化する。

	Args:
		hdf5_file_path (str): 解析対象のHDF5ファイルパス。
		n_fft (int): STFTのFFTサイズ。
		hop_length (int): STFTのホップ長。
		win_length (int): STFTの窓長。
		num_downsampling_layers (int): モデルのU-Net部分のダウンサンプリング層の数。
		output_dir (str): プロット画像などの出力先ディレクトリ。
	"""
	# ----- HDF5ファイルの読み込み -----
	try:
		hdf5_file = h5py.File(hdf5_file_path, 'r')
		print(f"✅ HDF5ファイル '{hdf5_file_path}' を読み込みました。")
	except FileNotFoundError:
		print(f"❌ エラー: ファイルが見つかりません: {hdf5_file_path}")
		return
	except Exception as e:
		print(f"❌ ファイル読み込み中にエラーが発生しました: {e}")
		return

	file_keys = list(hdf5_file.keys())
	if not file_keys:
		print("⚠️ ファイル内にデータが見つかりませんでした。処理を終了します。")
		hdf5_file.close()
		return

	print(f"  総ファイル数: {len(file_keys)}")

	# ----- 全ファイルのデータ集計 -----
	print("\n全ファイルのグラフ構造から **ノイズありグラフ** の入次数と誤差を集計中...")

	node_in_degree_counter = Counter()  # ★ ノイズありグラフの入次数のみカウント
	total_edges = 0  # ★ ノイズありグラフのエッジ数のみカウント
	max_node_index = -1
	total_node_error_sum = Counter()  # 誤差合計用 Counter
	node_occurrence_count = Counter()  # 出現回数用 Counter

	for file_key in tqdm(file_keys, desc="ファイル処理中"):
		data = load_data_from_hdf5(hdf5_file, file_key)
		if data:
			# --- 入次数集計 (ノイズありグラフのみ) ---
			noisy_targets = data['noisy_index'][:, 1]
			# clean_targets = data['clean_index'][:, 1] # clean_index は使用しない

			node_in_degree_counter.update(noisy_targets)  # ★ noisy_targets のみ更新
			# node_in_degree_counter.update(clean_targets) # ★ コメントアウト

			total_edges += len(noisy_targets)  # ★ noisy_targets の長さのみ加算
			# total_edges += len(noisy_targets) + len(clean_targets) # ★ 修正

			# --- 誤差集計 (変更なし) ---
			error_node_feat = data['error_node']
			mean_abs_error_per_node = np.mean(np.abs(error_node_feat), axis=1)
			node_indices = np.arange(len(mean_abs_error_per_node))

			for idx, error_val in zip(node_indices, mean_abs_error_per_node):
				total_node_error_sum[idx] += error_val
				node_occurrence_count[idx] += 1

			# 最大ノードインデックス更新 (noisy_indexのみ考慮)
			current_max_node_in_file = -1
			if len(noisy_targets) > 0:
				current_max_node_in_file = max(current_max_node_in_file, noisy_targets.max())
			# error_nodeのインデックスも考慮
			current_max_node_in_file = max(current_max_node_in_file,
			                               len(mean_abs_error_per_node) - 1 if len(mean_abs_error_per_node) > 0 else -1)
			max_node_index = max(max_node_index, current_max_node_in_file)

	print(f"集計完了。総エッジ数 (ノイズありグラフのみ): {total_edges}, 最大ノードインデックス: {max_node_index}")

	hdf5_file.close()  # ファイルを閉じる
	print(f"✅ HDF5ファイル '{hdf5_file_path}' を閉じました。")

	# ----- 集計結果から相関分析用のデータを作成 -----
	if total_edges > 0 and max_node_index >= 0:
		valid_node_indices = sorted(list(node_occurrence_count.keys()))  # 出現したノードのリスト

		if not valid_node_indices:
			print("\n⚠️ 有効なノードデータが見つかりませんでした。相関分析をスキップします。")
			return

		average_node_error = np.array([total_node_error_sum[idx] / node_occurrence_count[idx] for idx in valid_node_indices])
		node_degrees = np.array([node_in_degree_counter.get(idx, 0) for idx in valid_node_indices])  # Counterから取得

		correlation_df = pd.DataFrame({
			'Node Index': valid_node_indices,
			'Average Abs Error': average_node_error,
			'In-Degree Count (Noisy Graph)': node_degrees  # ★ 列名を変更
		})

		# --- 相関分析 ---
		# 接続回数が0のノードを除外して相関を計算 (0が多いと相関が見えにくくなるため)
		correlation_df_filtered = correlation_df[correlation_df['In-Degree Count (Noisy Graph)'] > 0]
		if len(correlation_df_filtered) < 2:
			print("\n⚠️ 接続回数が0より大きいノードが2つ未満のため、相関係数を計算できません。")
			correlation = np.nan
			p_value = np.nan
		else:
			correlation, p_value = pearsonr(correlation_df_filtered['Average Abs Error'],
			                                correlation_df_filtered['In-Degree Count (Noisy Graph)'])

		print("\n--- 全体での誤差と **ノイズありグラフ** の入次数の相関分析 ---")  # ★ タイトル変更
		print(f"  対象ノード数 (少なくとも1回出現): {len(correlation_df)}")
		print(f"  相関係数計算に使用したノード数 (入次数 > 0): {len(correlation_df_filtered)}")
		print(f"  ピアソンの相関係数 (r): {correlation:.4f}")
		print(f"  p値: {p_value:.4g}")

		if np.isnan(correlation):
			corr_strength = "計算不可"
		elif abs(correlation) >= 0.7:
			corr_strength = "強い相関"
		elif abs(correlation) >= 0.4:
			corr_strength = "中程度の相関"
		elif abs(correlation) >= 0.2:
			corr_strength = "弱い相関"
		else:
			corr_strength = "ほとんど相関なし"
		print(f"  相関の強さ: {corr_strength}")

		if np.isnan(p_value):
			print("  統計的有意性: 計算不可")
		elif p_value < 0.05:
			print("  統計的に有意な相関が見られます (p < 0.05)。")
		else:
			print("  統計的に有意な相関は見られません (p >= 0.05)。")

		# --- 全ノードの統計情報 ---
		all_counts_list = list(node_in_degree_counter.values())
		mean_count_all = np.mean(all_counts_list) if all_counts_list else 0
		variance_count_all = np.var(all_counts_list) if all_counts_list else 0
		std_dev_count_all = np.std(all_counts_list) if all_counts_list else 0

		print("\n--- 全ノードの接続回数（ノイズありグラフの入次数）統計 ---")  # ★ タイトル変更
		print(f"  平均接続回数: {mean_count_all:.4f}")
		print(f"  接続回数の分散: {variance_count_all:.4f}")
		print(f"  接続回数の標準偏差: {std_dev_count_all:.4f}")
		print("-" * 45)

		# --- 可視化 (散布図) ---
		plt.figure(figsize=(10, 8))
		scatter_plot = sns.scatterplot(data=correlation_df, x='Average Abs Error', y='In-Degree Count (Noisy Graph)', alpha=0.3,
		                               s=10)  # ★ Y軸ラベル変更
		# 回帰直線を追加 (オプション) - フィルタリング後のデータを使用
		if not correlation_df_filtered.empty:
			sns.regplot(data=correlation_df_filtered, x='Average Abs Error', y='In-Degree Count (Noisy Graph)', scatter=False,
			            color='red', line_kws={'linewidth': 1.5})

		plt.title('Overall Correlation between Node Error and In-Degree (Noisy Graph)')  # ★ タイトル変更
		plt.xlabel('Average Absolute Node Error (Feature Dimension Mean)')
		plt.ylabel('Total In-Degree Count (Noisy Graph Only)')  # ★ Y軸ラベル変更
		plt.grid(True, linestyle='--', alpha=0.6)

		# 画像ファイルとして保存
		output_plot_path = Path(output_dir) / "error_vs_indegree_correlation_noisy.png"  # ★ ファイル名変更
		output_plot_path.parent.mkdir(parents=True, exist_ok=True)  # 出力ディレクトリ作成
		try:
			plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
			print(f"✅ 相関プロットを '{output_plot_path}' に保存しました。")
		except Exception as e:
			print(f"❌ プロットの保存中にエラーが発生しました: {e}")

		# plt.show() # スクリプト実行時は不要な場合が多いのでコメントアウト

		# --- 相関データをCSVで保存 ---
		output_csv_path = Path(output_dir) / "error_vs_indegree_data_noisy.csv"  # ★ ファイル名変更
		try:
			correlation_df.to_csv(output_csv_path, index=False, float_format='%.6f')
			print(f"✅ 相関分析データを '{output_csv_path}' に保存しました。")
		except Exception as e:
			print(f"❌ CSVファイルの保存中にエラーが発生しました: {e}")

	elif total_edges == 0:
		print("\n⚠️ ノイズありグラフにエッジが見つからなかったため、相関分析を実行できません。")
	else:
		print("\n⚠️ 有効なノードデータが見つからなかったため、相関分析を実行できませんでした。")


# === スクリプト実行のためのメインブロック ===
if __name__ == "__main__":
	hdf5_path = f"/Users/a/Desktop/SpeqGAT_noise_reverb_analysis_results.h5"
	output_dir = f"{const.RESULT_DIR}/SpeqGAT/analysis_graph"

	parser = argparse.ArgumentParser(description="HDF5ファイルからノード誤差と入次数の相関を分析します。")
	parser.add_argument("--hdf5_file", type=str, default=hdf5_path, help="解析対象のHDF5ファイルパス")
	parser.add_argument("--n_fft", type=int, default=512, help="STFTのFFTサイズ")
	parser.add_argument("--hop_length", type=int, default=256, help="STFTのホップ長")
	parser.add_argument("--win_length", type=int, default=512, help="STFTの窓長")
	parser.add_argument("--num_downsampling", type=int, default=3, help="U-Netのダウンサンプリング層の数")
	parser.add_argument("--output_dir", type=str, default=output_dir, help="プロット画像とCSVの出力先ディレクトリ")

	args = parser.parse_args()

	# Pathオブジェクトを出力ディレクトリに使用
	output_directory = Path(args.output_dir)
	output_directory.mkdir(parents=True, exist_ok=True)  # ディレクトリ作成

	analyze_hdf5_correlation_noisy_graph(
		hdf5_file_path=args.hdf5_file,
		n_fft=args.n_fft,
		hop_length=args.hop_length,
		win_length=args.win_length,
		num_downsampling_layers=args.num_downsampling,
		output_dir=str(output_directory)  # 関数には文字列として渡す
	)
