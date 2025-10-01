# -*- coding: utf-8 -*-
"""
生成された音声ファイルパスとIR特徴量パスを整理し、
学習用データリストのCSVファイルを生成するスクリプト。

このスクリプトは、以下のファイルパスを紐づけてCSVに書き出します。
- 混合＋残響音声 (.wav)
- クリーン＋残響音声 (.wav)
- 元のクリーン音声 (.wav)
- IR特徴量ファイル (.npz)

使用例:
python format_complete_dataset.py \
    --mixed_dir "output_dataset/wav/mixed" \
    --clean_reverb_dir "output_dataset/wav/clean_reverb" \
    --clean_dir "path/to/original_clean" \
    --ir_dir "reverb_features" \
    --output_csv "dataset_list.csv"
"""

import os
import argparse
import csv
from collections import defaultdict
import glob
from pathlib import Path

# 既存のmymodule/my_func.pyをインポート
import sys

# このスクリプトがmymoduleと同じ階層にあることを前提
sys.path.append(os.path.join(os.path.dirname(__file__), 'mymodule'))
from mymodule import my_func


def format_complete_dataset_to_csv(mixed_dir, clean_reverb_dir, clean_dir, ir_dir, output_csv):
	"""
	指定されたディレクトリのファイルパスを整理し、CSVファイルとして保存します。

	Args:
		mixed_dir (str): 混合＋残響音声ファイルが格納されたディレクトリ。
		clean_reverb_dir (str): クリーン＋残響音声ファイルが格納されたディレクトリ。
		clean_dir (str): 元のクリーン音声ファイルが格納されたディレクトリ。
		ir_dir (str): IR特徴量ファイル（.npz）が格納されたディレクトリ。
		output_csv (str): 生成するCSVファイルのパス。
	"""
	# ファイルリストを取得
	mixed_files = my_func.get_file_list(mixed_dir, ext=".wav")
	clean_reverb_files = my_func.get_file_list(clean_reverb_dir, ext=".wav")
	clean_files = my_func.get_file_list(clean_dir, ext=".wav")
	ir_files = my_func.get_file_list(ir_dir, ext=".npz")

	# ファイル名をキーとしてパスを整理する辞書を作成
	mixed_dict = {my_func.get_file_name(f)[0]: f for f in mixed_files}
	clean_reverb_dict = {my_func.get_file_name(f)[0]: f for f in clean_reverb_files}
	clean_dict = {my_func.get_file_name(f)[0]: f for f in clean_files}
	ir_dict = {my_func.get_file_name(f)[0]: f for f in ir_files}

	# CSVヘッダー
	header = ["mixed_path", "clean_reverb_path", "clean_path", "ir_npz_path"]
	rows = []

	# mixed_dirのファイル名を基準に対応するパスを探す
	for mixed_name, mixed_path in mixed_dict.items():
		# ファイル名から元のクリーン音声名とIR名を抽出
		# フォーマット: speech_name_ir_name_snr.wav
		parts = mixed_name.split('_')
		if len(parts) >= 4:
			original_clean_name = parts[0]
			ir_name = parts[1]
			clean_reverb_name = f"{parts[0]}_{parts[1]}_clean_reverb"  # generate_complete_dataset.pyのファイル名フォーマットに合わせる
			ir_npz_name = ir_name  # generate_reverb_features.pyのファイル名フォーマットに合わせる

			clean_reverb_path = clean_reverb_dict.get(clean_reverb_name, "")
			clean_path = clean_dict.get(original_clean_name, "")
			ir_npz_path = ir_dict.get(ir_npz_name, "")

			# 対応するファイルが全て見つかった場合に行を追加
			if clean_reverb_path and clean_path and ir_npz_path:
				rows.append([mixed_path, clean_reverb_path, clean_path, ir_npz_path])
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
	parser.add_argument('--ir_dir', type=str, required=True,
	                    help='IR特徴量ファイル（.npz）が格納されたディレクトリ')
	parser.add_argument('--output_csv', type=str, default='dataset_list.csv',
	                    help='出力するCSVファイルのパス')

	args = parser.parse_args()

	format_complete_dataset_to_csv(
		mixed_dir=args.mixed_dir,
		clean_reverb_dir=args.clean_reverb_dir,
		clean_dir=args.clean_dir,
		ir_dir=args.ir_dir,
		output_csv=args.output_csv
	)