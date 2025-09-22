# -*- coding: utf-8 -*-
"""
クリーン音声、ノイズ、IRファイル（.npz形式）を組み合わせて、
残響のある学習データセットを生成するスクリプト。

IRファイルから複数の特徴量（RT60, ケプストラム係数など）を読み込み、
生成された音声とともにCSVファイルに記録します。

使用例:
python generate_complete_dataset.py \
    --speech_dir "path/to/clean_speech" \
    --noise_dir "path/to/noise" \
    --ir_dir "path/to/reverb_features" \
    --output_audio_dir "output_dataset/wav" \
    --output_csv "output_dataset/dataset_list.csv" \
    --snr 5
"""

import os
import random
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm
import argparse
import csv
import scipy

# 既存のmymodule/my_func.pyをインポート
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'mymodule'))
from mymodule import my_func
import config_generate_reverb_dataset as conf


def load_wav(filepath):
	"""WAVファイルを読み込み、波形データとサンプリングレートを返します。"""
	data, sr = sf.read(filepath)
	return data, sr


def save_wav(filepath, data, sr):
	"""波形データをWAVファイルとして保存します。"""
	sf.write(filepath, data, sr)


def random_crop(signal, target_length):
	"""信号を、指定された目標の長さにランダムに切り出します。"""
	if len(signal) <= target_length:
		repeat_times = int(np.ceil(target_length / len(signal)))
		signal = np.tile(signal, repeat_times)
	start = np.random.randint(0, len(signal) - target_length + 1)
	return signal[start: start + target_length]


def mix_snr(speech, noise, snr_db):
	"""音声とノイズを、指定されたSNR(dB)になるように混合します。"""
	speech_power = np.mean(speech ** 2)
	noise_power = np.mean(noise ** 2)
	target_noise_power = speech_power / (10 ** (snr_db / 10))
	noise_gain = np.sqrt(target_noise_power / (noise_power + 1e-10))
	adjusted_noise = noise * noise_gain
	return speech + adjusted_noise


def generate_complete_dataset(
		speech_dir,
		noise_dir,
		ir_dir,
		output_audio_dir,
		output_csv,
		snr=5
):
	"""
	クリーン音声、ノイズ、IRファイル（.npz）を組み合わせてデータセットを生成し、CSVに記録します。

	Args:
		speech_dir (str): クリーン音声ファイルが格納されたディレクトリのパス。
		noise_dir (str): ノイズファイルが格納されたディレクトリのパス。
		ir_dir (str): IR特徴量ファイル（.npz）が格納されたディレクトリのパス。
		output_audio_dir (str): 生成した音声ファイルを保存するディレクトリのパス。
		output_csv (str): 生成するデータリストCSVファイルのパス。
		snr (float, optional): 混合時のSNR(dB)。
	"""
	# 出力ディレクトリの作成
	my_func.make_dir(os.path.join(output_audio_dir, "mixed"))
	my_func.make_dir(os.path.join(output_audio_dir, "clean_reverb"))
	my_func.make_dir(output_csv)

	# ファイルリストを取得
	speech_list = my_func.get_file_list(speech_dir, ext=".wav")
	noise_list = my_func.get_file_list(noise_dir, ext=".wav")
	# IRは.npzファイルから読み込み
	ir_features_list = my_func.get_file_list(ir_dir, ext=".npz")

	if not speech_list or not noise_list or not ir_features_list:
		print("speech:", len(speech_list))
		print("noise:", len(noise_list))
		print("IR:", len(ir_features_list))
		raise ValueError("指定されたディレクトリに、必要なファイルが見つかりませんでした。")

	# CSVヘッダーの定義
	csv_header = ["mixed_path", "clean_reverb_path", "clean_path", "rir_path", "rt60", "c50", "d50", "cepstrum_coeffs"]
	csv_data = []

	print(f"Generating dataset with SNR={snr}dB...")
	for speech_file in tqdm(speech_list, desc="Processing files"):
		# クリーン音声ファイルをロード
		speech, sr = load_wav(speech_file)

		# ランダムにノイズとIR特徴量ファイルを選択
		noise_file = random.choice(noise_list)
		ir_features_file = random.choice(ir_features_list)

		noise, _ = load_wav(noise_file)
		ir_data = np.load(ir_features_file)
		# print(f"rt60: {ir_data['rt60']}, c50: {ir_data['c50']}, d50: {ir_data['d50']}")
		# sys.exit(2)

		# IRと特徴量を.npzファイルから読み込み
		signal_rir = ir_data['signal_rir']
		noise_rir = ir_data['noise_rir']

		# 残響特徴量を辞書にまとめる
		reverb_features = {
			'rt60': float(ir_data['rt60']),
			'c50': float(ir_data['c50']),
			'd50': float(ir_data['d50']),
			'cepstrum_coeffs': ir_data['cepstrum_coeffs'].tolist()  # np.ndarrayをリストに変換
		}

		# ノイズの長さをRIRの長さに合わせてクロップ
		noise_cropped = random_crop(noise, len(speech))

		# クリーン音声とノイズをIRで畳み込み
		reverb_speech = fftconvolve(speech, signal_rir, mode='full')[:len(speech)]
		reverb_noise = fftconvolve(noise_cropped, signal_rir, mode='full')[:len(noise_cropped)]

		# 音声の長さを合わせる
		min_len = min(len(reverb_speech), len(reverb_noise))
		reverb_speech = reverb_speech[:min_len]
		reverb_noise = reverb_noise[:min_len]
		reverb_speech = reverb_speech / np.max(np.abs(reverb_speech))
		reverb_noise = reverb_noise / np.max(np.abs(reverb_noise))

		# 残響ありの音声とノイズを混合
		mixed_reverb_audio = mix_snr(reverb_speech, reverb_noise, snr_db=snr)

		# 正規化して保存（-1.0〜1.0の範囲に収める）
		reverb_speech = reverb_speech / np.max(np.abs(reverb_speech))
		mixed_reverb_audio = mixed_reverb_audio / np.max(np.abs(mixed_reverb_audio))

		# ファイル名を生成
		speech_name = my_func.get_file_name(speech_file)[0]
		ir_name = my_func.get_file_name(ir_features_file)[0]

		mixed_filename = f"{speech_name}_{ir_name}_snr{snr}dB_mixed.wav"
		clean_reverb_filename = f"{speech_name}_{ir_name}_clean_reverb.wav"

		# 生成したファイルを保存
		mixed_path = os.path.join(output_audio_dir, "mixed", mixed_filename)
		clean_reverb_path = os.path.join(output_audio_dir, "clean_reverb", clean_reverb_filename)

		save_wav(mixed_path, mixed_reverb_audio, sr)
		save_wav(clean_reverb_path, reverb_speech, sr)
		# sys.exit(2)


		# CSVデータを作成
		csv_row = [mixed_path, clean_reverb_path, speech_file, ir_features_file, reverb_features['rt60'], reverb_features['c50'],
		           reverb_features['d50'], reverb_features['cepstrum_coeffs']]
		csv_data.append(csv_row)

	# CSVファイルへの書き込み
	with open(output_csv, 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(csv_header)
		writer.writerows(csv_data)

	print(f"✅ データセットの生成が完了しました。")
	print(f"✅ CSVファイルが '{output_csv}' に保存されました。")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='残響のある混合データセットを生成します。')
	parser.add_argument('--speech_dir', type=str, default=conf.speech_dir,
	                    help='クリーン音声ファイルが格納されたディレクトリ')
	parser.add_argument('--noise_dir', type=str, default=conf.noise_dir,
	                    help='ノイズファイルが格納されたディレクトリ')
	parser.add_argument('--ir_dir', type=str, default=conf.ir_dir,
	                    help='IR特徴量ファイル（.npz）が格納されたディレクトリ')
	parser.add_argument('--output_audio_dir', type=str, default=conf.output_audio_dir,
	                    help='生成した音声ファイルを保存するディレクトリ')
	parser.add_argument('--output_csv', type=str, default=conf.output_csv,
	                    help='生成するデータリストCSVファイルのパス')
	parser.add_argument('--snr', type=float, default=conf.snr,
	                    help='混合するSNR（dB）')

	args = parser.parse_args()

	generate_complete_dataset(
		speech_dir=args.speech_dir,
		noise_dir=args.noise_dir,
		ir_dir=args.ir_dir,
		output_audio_dir=args.output_audio_dir,
		output_csv=args.output_csv,
		snr=args.snr
	)