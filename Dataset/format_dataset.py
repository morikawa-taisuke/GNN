# -*- coding: utf-8 -*-
"""
生成された音声ファイルパスを整理し、学習用のCSVファイルを生成するスクリプト。
このスクリプトは、混合音声、クリーン残響音声、元のクリーン音声のパスを紐づけ、
一つのCSVファイルにまとめます。

使用例:
python format_dataset.py --mixed_dir "mixed_reverb" --clean_reverb_dir "clean_reverb" --clean_dir "original_clean" --output_csv "dataset_list.csv"
"""

import os
import argparse
import csv
from collections import defaultdict

# 既存のmymodule/my_func.pyをインポート
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'mymodule'))
from mymodule import my_func


def format_dataset_to_csv(mixed_dir, clean_reverb_dir, clean_dir, output_csv):
	"""
	指定されたディレクトリのファイルパスを整理し、CSVファイルとして保存します。

	Args:
		mixed_dir (str): 混合＋残響音声ファイルが格納されたディレクトリ。
		clean_reverb_dir (str): クリーン＋残響音声ファイルが格納されたディレクトリ。
		clean_dir (str): 元のクリーン音声ファイルが格納されたディレクトリ。
		output_csv (str): 生成するCSVファイルのパス。
	"""
	# ファイルリストを取得
	mixed_files = my_func.get_file_list(mixed_dir, ext=".wav")
	clean_reverb_files = my_func.get_file_list(clean_reverb_dir, ext=".wav")

	# ファイル名をキーとしてパスを整理する辞書
	mixed_dict = {my_func.get_file_name(f)[0]: f for f in mixed_files}
	clean_reverb_dict = {my_func.get_file_name(f)[0]: f for f in clean_reverb_files}
	clean_files = my_func.get_file_list(clean_dir, ext=".wav")
	clean_dict = {my_func.get_file_name(f)[0]: f for f in clean_files}

	# CSVヘッダー
	header = ["mixed_path", "clean_reverb_path", "clean_path"]
	rows = []

	# mixed_dirのファイル名を基準に対応するパスを探す
	for mixed_name, mixed_path in mixed_dict.items():
		# ファイル名から元のクリーン音声名を抽出
		# フォーマット: speech_name_noise_name_ir_name_snr.wav
		parts = mixed_name.split('_')
		if len(parts) >= 4:
			original_clean_name = parts[0]
			clean_reverb_name = f"{parts[0]}_{parts[2]}"  # 例: speech_name_ir_name

			clean_reverb_path = clean_reverb_dict.get(clean_reverb_name, "")
			clean_path = clean_dict.get(original_clean_name, "")

			# 対応するファイルが全て見つかった場合に行を追加
			if clean_reverb_path and clean_path:
				rows.append([mixed_path, clean_reverb_path, clean_path])
			else:
				print(f"Warning: Corresponding files not found for {mixed_name}. Skipping.")

	# CSVファイルへの書き込み
	if rows:
		my_func.make_dir(output_csv)
		with open(output_csv, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(header)
			writer.writerows(rows)
		print(f"✅ データセットのCSVファイルが '{output_csv}' に正常に作成されました。")
	else:
		print("Warning: No matching file triplets found. No CSV file was generated.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='音声ファイルから学習用CSVを生成します。')
	parser.add_argument('--mixed_dir', type=str, required=True,
	                    help='混合＋残響音声ファイルが格納されたディレクトリ')
	parser.add_argument('--clean_reverb_dir', type=str, required=True,
	                    help='クリーン＋残響音声ファイルが格納されたディレクトリ')
	parser.add_argument('--clean_dir', type=str, required=True,
	                    help='元のクリーン音声ファイルが格納されたディレクトリ')
	parser.add_argument('--output_csv', type=str, default='dataset_list.csv',
	                    help='出力するCSVファイルのパス')

	args = parser.parse_args()

	format_dataset_to_csv(
		mixed_dir=args.mixed_dir,
		clean_reverb_dir=args.clean_reverb_dir,
		clean_dir=args.clean_dir,
		output_csv=args.output_csv
	)