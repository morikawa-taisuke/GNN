"""
ミュージカルノイズの発生に関する性能評価をするプログラム

このモジュールは、特定の命名規則とメタデータを用いてオーディオファイルからメトリクスを照合、分析、抽出する機能を提供します。
音楽ノイズメトリクスの計算、バッチディレクトリの処理、CSV出力の生成を行います。

本モジュールは、クリーンなオーディオおよびモデル生成オーディオ出力を処理し、オーディオ分析に基づくメトリクスの計算と結果の要約を目的としています。

機能:
- get_file_id: 命名規則に基づきファイル名から一意の識別子を抽出します。
- parse_metadata: 指定されたファイル名から分析に有用なメタデータフィールドを抽出します。
- get_musical_noise_metrics: クリーン音声ファイルと推定音声ファイルに基づき、
  帯域特異的な音楽的ノイズ指標を計算します。
- batch_analyze_with_id_matching: クリーン音声ディレクトリとモデル音声ディレクトリ間でファイルを照合し、
  バッチ分析を実行し、結果をCSVファイルに出力します。"""
import os
import glob
import numpy as np
import librosa
import pandas as pd
from scipy.stats import kurtosis
from skimage import measure
from sympy.physics.units.definitions.unit_definitions import oersted
from tqdm import tqdm
import re


def get_file_id(filename):
	"""
	ファイル名から '話者番号_発話番号' を抽出してキーを返す
	例: '0001_001_...' -> '0001_001'
	"""
	parts = filename.split('_')
	if len(parts) >= 2:
		return f"{parts[0]}_{parts[1]}"
	return None


def parse_metadata(filename):
	"""
	ファイル名から分析に役立つメタデータを抽出する
	"""
	parts = filename.split('_')
	# 命名規則に基づきインデックスで取得（多少の形式ズレにも耐えられるよう考慮）
	metadata = {
		'NoiseType': parts[2] if len(parts) > 2 else 'unknown',
		'SNR': parts[4] if len(parts) > 4 else 'unknown',
	}
	return metadata


def get_musical_noise_metrics(y_clean, y_est, sr=16000):
	"""帯域別の指標計算（前回のロジックを継承）"""
	S_clean = np.abs(librosa.stft(y_clean, n_fft=512, hop_length=160))
	S_est = np.abs(librosa.stft(y_est, n_fft=512, hop_length=160))
	error_spec = np.abs(S_clean - S_est)

	freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
	bands = {'Low': (0, 1000), 'Mid': (1000, 4000), 'High': (4000, sr // 2)}

	band_results = {}
	for name, (f_min, f_max) in bands.items():
		idx = np.where((freqs >= f_min) & (freqs < f_max))[0]
		if len(idx) == 0: continue

		b_err = error_spec[idx, :]
		kurt = kurtosis(b_err.flatten())
		thresh = np.mean(b_err) + 2 * np.std(b_err)
		islands = measure.label((b_err > thresh).astype(int), connectivity=2).max()
		flux_var = np.var(np.diff(b_err, axis=1) ** 2)

		band_results[name] = {
			'Kurtosis': kurt,
			'IslandCount': islands,
			'FluxVar': flux_var
		}
	return band_results


def batch_analyze_with_id_matching(clean_dir, model_dirs, output_csv="detailed_analysis.csv"):
	"""
	ID（話者_発話）でマッチングして一括解析
	"""
	# 1. クリーン音声のリストを作成 (IDをキーにする)
	clean_files = glob.glob(os.path.join(clean_dir, "*.wav"))
	clean_map = {get_file_id(os.path.basename(f)): f for f in clean_files if get_file_id(os.path.basename(f))}

	all_results = []

	# 2. 各モデルディレクトリを走査
	for model_name, m_dir in model_dirs.items():
		print(f"Analyzing model: {model_name}...")
		est_files = glob.glob(os.path.join(m_dir, "*.wav"))

		for est_path in tqdm(est_files):
			filename = os.path.basename(est_path)
			file_id = get_file_id(filename)

			# クリーン音声とのマッチング確認
			if file_id not in clean_map:
				continue

			clean_path = clean_map[file_id]
			meta = parse_metadata(filename)

			# 音声読み込み
			y_clean, sr = librosa.load(clean_path, sr=16000)
			y_est, _ = librosa.load(est_path, sr=sr)

			# 長さ調整
			min_l = min(len(y_clean), len(y_est))
			metrics = get_musical_noise_metrics(y_clean[:min_l], y_est[:min_l], sr)

			# 帯域ごとの結果を保存
			for band, values in metrics.items():
				row = {
					'FileID': file_id,
					'Model': model_name,
					'Band': band,
					'NoiseType': meta['NoiseType'],
					'SNR': meta['SNR']
				}
				row.update(values)
				all_results.append(row)

	# 3. CSV出力
	if not all_results:
		print("\n[エラー] 解析結果が空です。以下の点を確認してください：")
		print(f"  - clean_dir のパスが正しいか: {clean_dir}")
		print(f"  - model_dirs のパスが正しいか: {list(model_dirs.values())}")
		print("  - ファイル名の命名規則（話者番号_発話番号）が一致しているか")
		return pd.DataFrame()  # 空のDFを返す

	df = pd.DataFrame(all_results)
	df.to_csv(output_csv, index=False)

	print(f"\n--- Analysis Complete ({len(df)} rows processed) ---")

	# 列が存在するか確認してから集計する
	if 'Model' in df.columns and 'Band' in df.columns:
		summary = df.groupby(['Model', 'Band'])[['Kurtosis', 'IslandCount', 'FluxVar']].mean()
		print(summary)
	else:
		print("警告: 必要な列が生成されませんでした。")

	return df


# 設定例
if __name__ == "__main__":
	# 実際のパスに合わせて書き換えてください
	edge_aria_list = ["temporal", "all"]   # all, temporal
	edge_select_list = ["knn", "random"] # knn, random

	wave_type_list = ['noise_only', 'reverb_only', 'noise_reverb']

	for edge_aria in edge_aria_list:
		for edge_select in edge_select_list:
			for wave_type in wave_type_list:
				target_model_dirs = {
					'GCN': f'/Users/a/Documents/sound_data/RESULT/output_wav/Random_Dataset_VCTK_DEMAND_1ch/SpeqGCN/SpeqGCN_{wave_type}_32node_{edge_aria}_{edge_select}',
					'GAT': f'/Users/a/Documents/sound_data/RESULT/output_wav/Random_Dataset_VCTK_DEMAND_1ch/SpeqGAT/SpeqGAT_{wave_type}_32node_{edge_aria}_{edge_select}'
				}

				df = batch_analyze_with_id_matching(
					clean_dir='/Users/a/Documents/sound_data/mix_data/Random_Dataset_1ch/clean',
					model_dirs=target_model_dirs,
					output_csv=f"SpeqGNN_{wave_type}_{edge_aria}_{edge_select}.csv"
				)
