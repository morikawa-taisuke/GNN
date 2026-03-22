import numpy as np
import os
import csv
import argparse
import torch
from tqdm import tqdm

# 自作モジュール
# パスが通ってぁE��ことを前提としてぁE��ぁE
from evaluation.PESQ import pesq_evaluation
from evaluation.STOI import stoi_evaluation
from evaluation.SI_SDR import sisdr_evaluation
from mymodule import my_func, const


def main(input_csv_path, target_column, estimation_column, estimation_dir, out_path):
	"""客観評価をCSV入力で行う"""
	print(f"Input CSV: {input_csv_path}")
	print(f"Target Column: {target_column}")
	print(f"Estimation Column: {estimation_column}")
	print(f"Estimation Directory: {estimation_dir}")
	print(f"Output CSV: {out_path}")

	# --- 1. 入力CSVからファイルペアを読み込む ---
	file_pairs = []
	try:
		with open(input_csv_path, "r", encoding="utf-8") as f:
			reader = csv.reader(f)
			header = next(reader)
			try:
				target_idx = header.index(target_column)
				estimation_idx = header.index(estimation_column)
			except ValueError as e:
				print(f"❁Eエラー: CSVヘッダーに持E��された列が見つかりません '{header}': {e}")
				return

			for row in reader:
				if len(row) > max(target_idx, estimation_idx):
					target_file = row[target_idx]
					# estimationファイルはファイル名だけ取征E
					estimation_filename_in_csv = row[estimation_idx]

					if target_file and estimation_filename_in_csv:
						# estimationファイルのフルパスを生戁E
						estimation_base_name = os.path.basename(estimation_filename_in_csv)
						estimation_file = os.path.join(estimation_dir, estimation_base_name)

						if os.path.exists(target_file) and os.path.exists(estimation_file):
							file_pairs.append((target_file, estimation_file))
						else:
							print(
								f"⚠�E�E警呁E 行をスキチE�Eします。ファイルパスが存在しなぁE��、ファイルが見つかりません: target='{target_file}', estimation='{estimation_file}'")
					else:
						print(f"⚠�E�E警呁E CSV冁E�Eパスが空です。行をスキチE�EしまぁE {row}")
				else:
					print(f"⚠�E�E警呁E 不正な形式�E行をスキチE�EしまぁE {row}")

	except FileNotFoundError:
		print(f"❁Eエラー: 入力CSVファイルが見つかりません: {input_csv_path}")
		return

	if not file_pairs:
		print("評価対象の有効なファイルペアが見つかりませんでした、E)
		return

	# --- 2. 出力ファイルを作�Eし、�EチE��ーを書き込む ---
	output_dir = os.path.dirname(out_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	with open(out_path, "w", encoding="utf-8", newline="") as f:
		# メタ惁E��の書き込み
		f.write(f"input_csv,{input_csv_path}\n")
		f.write(f"target_column,{target_column}\n")
		f.write(f"estimation_column,{estimation_column}\n")
		f.write(f"estimation_dir,{estimation_dir}\n")
		# CSVヘッダーの書き込み
		f.write("target_name,estimation_name,pesq,stoi,sisdr\n")

	# --- 3. 評価持E���E初期匁E---
	pesq_scores = []
	stoi_scores = []
	sisdr_scores = []
	evaluated_count = 0

	# --- 4. 吁E��ァイルペアを�E琁E---
	for target_file, estimation_file in tqdm(file_pairs, desc="Evaluating files"):
		try:
			target_name, _ = my_func.get_file_name(target_file)
			estimation_name, _ = my_func.get_file_name(estimation_file)

			target_data, _ = my_func.load_wav(target_file)
			estimation_data, _ = my_func.load_wav(estimation_file)

			min_length = min(len(target_data), len(estimation_data))
			target_data = target_data[:min_length]
			estimation_data = estimation_data[:min_length]

			target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
			estimation_data = np.nan_to_num(estimation_data, nan=0.0, posinf=0.0, neginf=0.0)

			pesq_score = pesq_evaluation(target_data, estimation_data)
			stoi_score = stoi_evaluation(target_data, estimation_data)
			sisdr_score = sisdr_evaluation(target_data, estimation_data)

			pesq_scores.append(pesq_score)
			stoi_scores.append(stoi_score)
			sisdr_scores.append(sisdr_score.item() if torch.is_tensor(sisdr_score) else sisdr_score)
			evaluated_count += 1

			with open(out_path, "a", encoding="utf-8", newline="") as f:
				writer = csv.writer(f)
				writer.writerow([target_name, estimation_name, pesq_score, stoi_score, sisdr_score.item()])

		except Exception as e:
			print(f"ファイル処琁E��にエラーが発生しました {target_file}, {estimation_file}: {e}")

	# --- 5. 平坁E��を計算して書き込む ---
	if len(pesq_scores) > 0:
		pesq_ave, pesq_var = np.mean(pesq_scores), np.var(pesq_scores)
		stoi_ave, stoi_var = np.mean(stoi_scores), np.var(stoi_scores)
		sisdr_ave, sisdr_var = np.mean(sisdr_scores), np.var(sisdr_scores)

		with open(out_path, "a", encoding="utf-8", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["average", "", pesq_ave, stoi_ave, sisdr_ave])
			writer.writerow(["variance", "", pesq_var, stoi_var, sisdr_var])

		print("\n--- スコア統訁E---")
		print(f"PESQ   : Ave {pesq_ave:.3f}, Var {pesq_var:.3f}")
		print(f"STOI   : Ave {stoi_ave:.3f}, Var {stoi_var:.3f}")
		print(f"SI-SDR : Ave {sisdr_ave:.3f}, Var {sisdr_var:.3f}")
	# ... 後略 ...
	else:
		print("評価されたファイルはありませんでした、E)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="""CSVファイルに基づぁE��音声評価�E�EESQ, STOI, SI-SDR�E�を実行します、ECSVファイルには、E��声ファイルへのパスを含む列が忁E��です、E"",
		formatter_class=argparse.RawTextHelpFormatter
	)

	dir_name = "Random_Dataset_VCTK_DEMAND_1ch"
	model_type = "SpeqUNet"
	estimation_column_list = [
		"noise_only",
		"reverb_only",
		"noise_reverb"
	]
	for estimation_column in estimation_column_list:
		input_csv = f"\\\\192.168.11.63\\Shared_NAS\\morikawa\\saound_file\\mix_data\\Random_Dataset_VCTK_DEMAND_1ch\\test.csv"
		# out_name = f"SISDR_{model_type}_{estimation_column}_32node_temporal_knn"  # 出力名
		out_name = f"{model_type}_{estimation_column}"  # 出力名
		estimation_dir = f"\\\\192.168.11.63\\io-data_nas\\morikawa\\RESULT\\output_wav\\Random_Dataset_VCTK_DEMAND_1ch\\{model_type}\\{model_type}_{estimation_column}"
		out_path = f"\\\\192.168.11.63\\io-data_nas\\morikawa\\RESULT\\evaluation\\Random_Dataset_VCTK_DEMAND_1ch\\{model_type}\\{model_type}_{estimation_column}.csv"

		# parser.add_argument("--input_csv", type=str, default=input_csv, help="入力CSVファイルのパス。侁E 'C:/data/test.csv'")
		# parser.add_argument("--target_column", type=str, default="clean",
		# 					help="ターゲチE���E�クリーンな�E�音声ファイルへのパスを含む列�E名前、EチE��ォルチE 'clean')")
		# parser.add_argument("--estimation_column", type=str, default=estimation_column,
		# 					help="評価対象のファイル名が含まれる列�E名前。侁E 'noise_only'")
		# parser.add_argument("--estimation_dir", type=str, default=estimation_dir,
		# 					help="評価対象の音声ファイルが格納されてぁE��チE��レクトリのパス、E)
		# parser.add_argument("--out_path", type=str, default=out_path,
		# 					help="出力評価CSVファイルを保存するパス。侁E 'C:/results/evaluation.csv'")
		#
		# args = parser.parse_args()

		# main(
		# 	input_csv_path=args.input_csv,
		# 	target_column=args.target_column,
		# 	estimation_column=args.estimation_column,
		# 	estimation_dir=args.estimation_dir,
		# 	out_path=args.out_path,
		# )
		main(
			input_csv_path=input_csv,
			target_column="clean",
			estimation_column=estimation_column,
			estimation_dir=estimation_dir,
			out_path=out_path,
		)


	# --input_csv
	# C:\Users\kataoka-lab\Desktop\sound_data\mix_data\Random_Dataset_VCTK_DEMAND_1ch\test.csv
	# --target_column
	# clean
	# --estimation_column
	# noise_reverb
	# --estimation_dir
	# \\\\192.168.11.63\io-data_nas\morikawa\RESULT\output_wav\Random_Dataset_VCTK_DEMAND_1ch\AAA_SpeqGAT\AAA_SpeqGAT_noise_only_32node_temporal_random
	# --out_path
	# \\\\192.168.11.63\io-data_nas\morikawa\RESULT\evaluation\Random_Dataset_VCTK_DEMAND_1ch\AAA_SpeqGATAAA_SpeqGAT_noise_only_32node_temporal_random.csv

