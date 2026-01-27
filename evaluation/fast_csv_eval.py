import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
# 評価ライブラリ
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

# SI-SDRは既存のモジュールまたは計算式を使用
from evaluation.SI_SDR import sisdr_evaluation
from mymodule import my_func, const


def process_single_pair(target_file, estimation_file, fs, pesq_mode):
	"""
	1つのファイルペアに対して客観評価を実行するワーカー関数
	"""
	try:
		target_name, _ = my_func.get_file_name(target_file)
		estimation_name, _ = my_func.get_file_name(estimation_file)

		# 音源の読み込み
		target_data, sr_t = my_func.load_wav(target_file)
		estimation_data, sr_e = my_func.load_wav(estimation_file)

		# サンプリングレートの整合性チェック
		if sr_t != fs or sr_e != fs:
			# 必要に応じてリサンプリング等の処理を追加可能
			pass

		# 長さの調整
		min_length = min(len(target_data), len(estimation_data))
		target_data = target_data[:min_length]
		estimation_data = estimation_data[:min_length]

		# 異常値(NaN/Inf)の除去
		target_data = np.nan_to_num(target_data)
		estimation_data = np.nan_to_num(estimation_data)

		# --- 客観評価の計算 ---

		# 1. PESQ (pesqライブラリを使用)
		# fsは16000か8000である必要があります
		pesq_score = pesq(fs, target_data, estimation_data, pesq_mode)

		# 2. STOI (pystoiを使用)
		stoi_score = stoi(target_data, estimation_data, fs, extended=False)

		# 3. SI-SDR
		sisdr_score = sisdr_evaluation(target_data, estimation_data)
		if hasattr(sisdr_score, 'item'):
			sisdr_score = sisdr_score.item()

		return [target_name, estimation_name, pesq_score, stoi_score, sisdr_score]

	except Exception as e:
		return f"Error in {os.path.basename(target_file)}: {str(e)}"


def main(input_csv, target_column, estimation_column, estimation_dir, out_path):
	# parser.add_argument("--input_csv", type=str, default=input_csv)
	# parser.add_argument("--target_column", type=str, default="clean")
	# parser.add_argument("--estimation_column", type=str, default=estimation_column)
	# parser.add_argument("--estimation_dir", type=str, default=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}")
	# parser.add_argument("--out_path", type=str, default=f"{const.EVALUATION_DIR}/{dir_name}/{model_type}/{out_name}.csv")

	# 評価設定
	fs = const.SR
	mode = "nb" # wb, nb
	workers = None

	# --- ファイルリストの準備 ---
	"""file_pairs = []
	if not os.path.exists(input_csv):
		print(f"Input CSV not found: {input_csv}")
		return

	with open(input_csv, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			target_file = row[target_column]
			est_filename = os.path.basename(row[estimation_column])
			estimation_file = os.path.join(estimation_dir, est_filename)

			if os.path.exists(target_file) and os.path.exists(estimation_file):
				file_pairs.append((target_file, estimation_file))

	if not file_pairs:
		print("No valid file pairs found.")
		return"""
	file_pairs = []
	try:
		with open(input_csv, "r", encoding="utf-8") as f:
			reader = csv.reader(f)
			header = next(reader)
			try:
				target_idx = header.index(target_column)
				estimation_idx = header.index(estimation_column)
			except ValueError as e:
				print(f"❌ エラー: CSVヘッダーに指定された列が見つかりません '{header}': {e}")
				return

			for row in reader:
				if len(row) > max(target_idx, estimation_idx):
					target_file = row[target_idx]
					# estimationファイルはファイル名だけ取得
					estimation_filename_in_csv = row[estimation_idx]

					if target_file and estimation_filename_in_csv:
						# estimationファイルのフルパスを生成
						estimation_base_name = os.path.basename(estimation_filename_in_csv)
						# print(estimation_filename_in_csv)
						# print(estimation_base_name)
						# exit()
						estimation_file = os.path.join(estimation_dir, estimation_base_name)

						if os.path.exists(target_file) and os.path.exists(estimation_file):
							file_pairs.append((target_file, estimation_file))
						else:
							print(
								f"⚠️ 警告: 行をスキップします。ファイルパスが存在しないか、ファイルが見つかりません: target='{target_file}', estimation='{estimation_file}'")
					else:
						print(f"⚠️ 警告: CSV内のパスが空です。行をスキップします: {row}")
				else:
					print(f"⚠️ 警告: 不正な形式の行をスキップします: {row}")

	except FileNotFoundError:
		print(f"❌ エラー: 入力CSVファイルが見つかりません: {input_csv}")
		return

	if not file_pairs:
		print("評価対象の有効なファイルペアが見つかりませんでした。")
		return

	# --- 並列実行 ---
	results = []
	print(f"Evaluating {len(file_pairs)} files on CPU ({fs}Hz, {mode})...")

	with ProcessPoolExecutor(max_workers=workers) as executor:
		futures = [executor.submit(process_single_pair, t, e, fs, mode) for t, e in file_pairs]
		for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
			res = future.result()
			if isinstance(res, list):
				results.append(res)
			else:
				print(f"\n{res}")

	if not results: return

	# --- 統計計算と保存 ---
	res_np = np.array([r[2:] for r in results], dtype=float)
	aves = np.mean(res_np, axis=0)
	vars = np.var(res_np, axis=0)

	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["PESQ_Mode", mode, "Sampling_Rate", fs])
		writer.writerow(["target_name", "estimation_name", "pesq", "stoi", "sisdr"])
		writer.writerows(results)
		writer.writerow([])
		writer.writerow(["average", "", *aves])
		writer.writerow(["variance", "", *vars])

	print(f"\nSaved to: {out_path}")
	print(f"Ave: PESQ={aves[0]:.3f}, STOI={aves[1]:.3f}, SI-SDR={aves[2]:.3f}")


if __name__ == "__main__":
	# 実際のパスに合わせて書き換えてください
	model_list = [
		"UGCN",
		"UGAT",
		"SpeqGCN",
		"SpeqGAT"
	]
	edge_aria_list = ["temporal", "all"]  # all, temporal
	edge_select_list = ["knn", "random"]  # knn, random
	wave_type_list = ['noise_only', 'reverb_only', 'noise_reverb']
	dir_name = "Random_Dataset_VCTK_DEMAND_1ch"
	num_node = 32

	for model_type in model_list:
		for edge_aria in edge_aria_list:
			for edge_select in edge_select_list:
				for wave_type in wave_type_list:
					out_name = f"{model_type}_{wave_type}_{num_node}node_{edge_aria}_{edge_select}"  # 出力名
					input_csv = f"{const.MIX_DATA_DIR}/{dir_name}/test.csv"
					# UGAT_noise_only_32node_all_knn
					estimation_dir = f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}"
					out_path = f"{const.EVALUATION_DIR}/{dir_name}_nb/{model_type}/{out_name}_nb.csv"

					target_column = "clean"
					estimation_column = wave_type
					print(estimation_dir)
					main(input_csv, target_column, estimation_column, estimation_dir, out_path)
