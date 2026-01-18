import numpy as np
import os
import csv
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 評価ライブラリ
from pesq import pesq
from pystoi import stoi
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


def main():
	parser = argparse.ArgumentParser(description="CPU-based Audio Evaluation using 'pesq' library")

	# パス設定 (既存の構成を継承)
	dir_name = "DEMAND_DEMAND"
	model_type = "ConvTasNet"
	input_csv = f"{const.MIX_DATA_DIR}/{dir_name}/test.csv"
	estimation_column = "reverb_only"
	out_name = f"{model_type}_{estimation_column}_cpu"

	parser.add_argument("--input_csv", type=str, default=input_csv)
	parser.add_argument("--target_column", type=str, default="clean")
	parser.add_argument("--estimation_column", type=str, default=estimation_column)
	parser.add_argument("--estimation_dir", type=str, default=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{model_type}/{out_name}")
	parser.add_argument("--out_path", type=str, default=f"{const.EVALUATION_DIR}/{dir_name}/{model_type}/{out_name}.csv")

	# 評価設定
	parser.add_argument("--fs", type=int, default=16000, help="Sampling rate (16000 or 8000)")
	parser.add_argument("--mode", type=str, default="wb", choices=["wb", "nb"],
	                    help="PESQ mode: 'wb' (wideband) or 'nb' (narrowband)")
	parser.add_argument("--workers", type=int, default=None, help="Number of CPU cores")

	args = parser.parse_args()

	# --- ファイルリストの準備 ---
	file_pairs = []
	if not os.path.exists(args.input_csv):
		print(f"Input CSV not found: {args.input_csv}")
		return

	with open(args.input_csv, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			target_file = row[args.target_column]
			est_filename = os.path.basename(row[args.estimation_column])
			estimation_file = os.path.join(args.estimation_dir, est_filename)

			if os.path.exists(target_file) and os.path.exists(estimation_file):
				file_pairs.append((target_file, estimation_file))

	if not file_pairs:
		print("No valid file pairs found.")
		return

	# --- 並列実行 ---
	results = []
	print(f"Evaluating {len(file_pairs)} files on CPU ({args.fs}Hz, {args.mode})...")

	with ProcessPoolExecutor(max_workers=args.workers) as executor:
		futures = [executor.submit(process_single_pair, t, e, args.fs, args.mode) for t, e in file_pairs]
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

	os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
	with open(args.out_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["PESQ_Mode", args.mode, "Sampling_Rate", args.fs])
		writer.writerow(["target_name", "estimation_name", "pesq", "stoi", "sisdr"])
		writer.writerows(results)
		writer.writerow([])
		writer.writerow(["average", "", *aves])
		writer.writerow(["variance", "", *vars])

	print(f"\nSaved to: {args.out_path}")
	print(f"Ave: PESQ={aves[0]:.3f}, STOI={aves[1]:.3f}, SI-SDR={aves[2]:.3f}")


if __name__ == "__main__":
	main()