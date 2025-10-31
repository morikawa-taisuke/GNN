import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import os


def extract_info(name):
	"""
	ファイル名から雑音名、SNR、残響時間を抽出する関数。
	形式: 「..._雑音名_..._SNRdB_..._残響時間msec_None」
	"""

	if not isinstance(name, str):
		return None, None, None

	# 残響時間を抽出 (例: 760msec)
	reverb_match = re.search(r'(\d+msec)', name)
	reverberation_time = reverb_match.group(1) if reverb_match else None

	# SNRを抽出 (例: -5db, 11db)
	snr_match = re.search(r'(-?\d+db)', name)
	snr = snr_match.group(1) if snr_match else None

	# 雑音名を抽出
	parts = name.split('_')
	noise = None

	if len(parts) > 3:  # pX_Y_..._None の形式を想定
		noise_parts = []
		# 話者番号(parts[0]), 発話番号(parts[1]),
		# 最後の'None'(parts[-1]) を除外
		for part in parts[2:-1]:
			is_info_part = False

			if reverb_match and part == reverb_match.group(1):
				is_info_part = True
			elif snr_match and part == snr_match.group(1):
				is_info_part = True
			elif re.match(r'^\d+ch$', part):  # チャンネル数 (例: 01ch)
				is_info_part = True

			if not is_info_part:
				noise_parts.append(part)

		if noise_parts:
			noise = '_'.join(noise_parts)

	return noise, snr, reverberation_time


# --- 1. 元のCSVファイルの読み込みと前処理 ---
print("--- スコアファイルの読み込みと前処理を開始 ---")

# 分析対象の元のCSVファイル
# (train.csv は訓練パス一覧のため、ここでは使用しません)
original_files = {
	'noise_only': '/Users/a/Documents/sound_data/RESULT/evalation/SISDR_SpeqGAT_noise_only_32node_temporal_knn_CSV.csv',
	'noise_reverb': '/Users/a/Documents/sound_data/RESULT/evalation/SISDR_SpeqGAT_noise_reverb_32node_temporal_knn_CSV.csv',
	'reverb_only': '/Users/a/Documents/sound_data/RESULT/evalation/SISDR_SpeqGAT_reverb_only_32node_temporal_knn_CSV.csv'
}

all_data = []
files_processed_count = 0

for task_name, filepath in original_files.items():
	try:
		# スコアCSVは先頭4行がヘッダー情報のためスキップ
		df = pd.read_csv(filepath, skiprows=4)

		# 'estimation_name' が存在するか確認
		if 'estimation_name' not in df.columns:
			print(f"警告: {filepath} に 'estimation_name' カラムがありません。スキップします。")
			continue

		# ファイル名解析を適用
		df[['Noise', 'SNR', 'ReverberationTime']] = df['estimation_name'].apply(
			lambda x: pd.Series(extract_info(x))
		)

		df['task'] = task_name  # タスク名（実験の種類）をカラムに追加
		all_data.append(df)
		files_processed_count += 1
		print(f"{filepath} を読み込み、解析しました。")

	except FileNotFoundError:
		print(f"警告: {filepath} が見つかりません。スキップします。")
	except Exception as e:
		print(f"警告: {filepath} の読み込み中にエラーが発生しました: {e}")

if files_processed_count == 0:
	print("エラー: 分析対象のスコアファイルが一つも見つかりませんでした。処理を中断します。")
else:
	# データをすべて結合
	df_combined = pd.concat(all_data, ignore_index=True)

	# --- 2. データ前処理 (数値化) ---

	# SNR (例: '-5db') を数値 (-5) に変換
	df_combined['SNR_value'] = pd.to_numeric(
		df_combined['SNR'].str.replace('db', ''),
		errors='coerce'  # 変換不能な値は NaN (Not a Number) にする
	)

	# ReverberationTime (例: '760msec') を数値 (760) に変換
	df_combined['Reverb_value'] = pd.to_numeric(
		df_combined['ReverberationTime'].str.replace('msec', ''),
		errors='coerce'
	)

	# Noise カラムの欠損値 (NaN や None) を 'No_Noise' というカテゴリ名に置き換える
	df_combined['Noise'] = df_combined['Noise'].fillna('No_Noise')
	df_combined['Noise'] = df_combined['Noise'].replace('None', 'No_Noise')

	print("\n--- 結合・前処理後のデータ確認 ---")
	print("Combined DataFrame Info:")
	df_combined.info()

	# --- 3. 分析と可視化 (グラフ作成) ---

	sns.set_style("whitegrid")

	# --- 分析1: 雑音の種類による性能差 (sisdr) ---
	print("\n--- 分析1: 雑音の種類による性能差 ---")

	# 'No_Noise' (reverb_onlyタスクなど) を除外したデータで平均値を計算
	noise_data = df_combined[df_combined['Noise'] != 'No_Noise']
	if not noise_data.empty:
		# 平均スコアでソートするための順序を取得
		noise_order = noise_data.groupby('Noise')['sisdr'].mean().sort_values(ascending=False).index

		plt.figure(figsize=(12, 7))
		sns.barplot(
			data=noise_data,
			x='Noise',
			y='sisdr',
			hue='task',  # 'noise_only' か 'noise_reverb' か
			order=noise_order
		)
		plt.title('Noise Type vs. Average SISDR Score (excl. No_Noise)', fontsize=16)
		plt.xlabel('Noise Type', fontsize=12)
		plt.ylabel('Average SISDR Score', fontsize=12)
		plt.xticks(rotation=45, ha='right')
		plt.legend(title='Task')
		plt.tight_layout()
		plt.show()
		# plt.savefig('noise_vs_sisdr.png')
		# print("グラフ 'noise_vs_sisdr.png' を保存しました。")
	else:
		print("雑音データ（'No_Noise'以外）が見つかりませんでした。")

	# --- 分析2: SNRの大きさによる性能差 (sisdr) ---
	print("\n--- 分析2: SNRの大きさによる性能差 ---")

	# SNRを含むデータ (NaN は除く)
	snr_data = df_combined.dropna(subset=['SNR_value'])
	if not snr_data.empty:
		plt.figure(figsize=(10, 6))
		sns.lineplot(
			data=snr_data,
			x='SNR_value',
			y='sisdr',
			hue='task',  # タスクごとに線を描画
			marker='o'
		)
		plt.title('SNR vs. Average SISDR Score', fontsize=16)
		plt.xlabel('SNR (db)', fontsize=12)
		plt.ylabel('Average SISDR Score', fontsize=12)
		plt.legend(title='Task')
		plt.tight_layout()
		plt.show()
		# plt.savefig('snr_vs_sisdr.png')
		# print("グラフ 'snr_vs_sisdr.png' を保存しました。")
	else:
		print("SNRデータが見つかりませんでした。")

	# --- 分析3: 残響時間による性能差 (sisdr) ---
	print("\n--- 分析3: 残響時間による性能差 ---")

	# 残響時間を含むデータ (NaN は除く)
	reverb_data = df_combined.dropna(subset=['Reverb_value'])
	if not reverb_data.empty:
		plt.figure(figsize=(10, 6))
		sns.scatterplot(
			data=reverb_data,
			x='Reverb_value',
			y='sisdr',
			hue='task',  # 'reverb_only' か 'noise_reverb' か
			alpha=0.7,
			s=50
		)
		plt.title('Reverberation Time vs. SISDR Score', fontsize=16)
		plt.xlabel('Reverberation Time (msec)', fontsize=12)
		plt.ylabel('SISDR Score', fontsize=12)
		plt.legend(title='Task')
		plt.tight_layout()
		plt.show()
		# plt.savefig('reverb_vs_sisdr.png')
		# print("グラフ 'reverb_vs_sisdr.png' を保存しました。")
	else:
		print("残響時間のデータが見つかりませんでした。")

	print("\nすべての分析プログラムが完了しました。")