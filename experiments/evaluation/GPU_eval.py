import numpy as np
import os
import csv
import argparse
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path

# torchmetricsの音声評価指標を使用
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

from mymodule import my_func, const, confirmation_GPU


def main(input_csv, estimation_dir, out_path, target_column="clean", estimation_column="noise_only"):
	parser = argparse.ArgumentParser(description="GPU-based Audio Evaluation using torchmetrics")

	parser.add_argument("--input_csv", type=str, default=input_csv)
	parser.add_argument("--estimation_dir", type=str, default=estimation_dir)
	parser.add_argument("--out_path", type=str, default=out_path)
	parser.add_argument("--target_column", type=str, default=target_column)
	parser.add_argument("--estimation_column", type=str, default=estimation_column)

	# 評価設定
	sampling_rate = const.SR
	mode = "wb"
	device =  confirmation_GPU.get_device()
	parser.add_argument("--mode", type=str, default="wb", choices=["wb", "nb"], help="PESQ mode: 'wb' or 'nb'")

	args = parser.parse_args()
	print(f"Using device: {device}")

	# --- メトリクスの初期化 ---
	# PESQとSTOIは内部的にCPUを使用する場合が多いですが、テンソルを直接渡せます
	pesq_metric = PerceptualEvaluationSpeechQuality(sampling_rate, mode).to(device)
	stoi_metric = ShortTimeObjectiveIntelligibility(sampling_rate, extended=False).to(device)
	sisdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

	# --- ファイルリストの準備 ---
	file_pairs = []
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

	# --- 評価実行 ---
	results = []
	pesq_list, stoi_list, sisdr_list = [], [], []

	print(f"Evaluating {len(file_pairs)} files on {device}...")

	# I/O高速化のため、ループ内ではリストへの蓄積のみ行う
	for target_path, est_path in tqdm(file_pairs, desc="Evaluating"):
		try:
			# 音源の読み込み (torchaudioを使用)
			target_wav, sr_t = torchaudio.load(target_path)
			est_wav, sr_e = torchaudio.load(est_path)

			# デバイスへ転送
			target_wav = target_wav.to(device)
			est_wav = est_wav.to(device)

			# 長さの調整 (最短に合わせる)
			min_len = min(target_wav.shape[-1], est_wav.shape[-1])
			target_wav = target_wav[..., :min_len]
			est_wav = est_wav[..., :min_len]

			# 指標の計算
			p_val = pesq_metric(est_wav, target_wav).item()
			s_val = stoi_metric(est_wav, target_wav).item()
			si_val = sisdr_metric(est_wav, target_wav).item()

			target_name = os.path.basename(target_path)
			est_name = os.path.basename(est_path)

			results.append([target_name, est_name, p_val, s_val, si_val])
			pesq_list.append(p_val)
			stoi_list.append(s_val)
			sisdr_list.append(si_val)

		except Exception as e:
			print(f"\nError processing {os.path.basename(target_path)}: {e}")

	# --- 統計計算 ---
	if not results: return

	aves = [np.mean(pesq_list), np.mean(stoi_list), np.mean(sisdr_list)]
	vars = [np.var(pesq_list), np.var(stoi_list), np.var(sisdr_list)]

	# --- 結果の一括書き込み ---
	os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
	with open(args.out_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["Device", args.device, "PESQ_Mode", mode, "FS", sampling_rate])
		writer.writerow(["target_name", "estimation_name", "pesq", "stoi", "sisdr"])
		writer.writerows(results)
		writer.writerow([])
		writer.writerow(["average", "", *aves])
		writer.writerow(["variance", "", *vars])

	print(f"\nSaved to: {args.out_path}")
	print(f"Ave: PESQ={aves[0]:.3f}, STOI={aves[1]:.3f}, SI-SDR={aves[2]:.3f}")


if __name__ == "__main__":
	estimation_column = "noise_only"
	bases_name = ""
	model_type = "Wave_UGNN"

	input_csv = const.MIX_DATA_DIR / bases_name / model_type / "test.csv"
	estimation_dir = const.OUTPUT_WAV_DIR / bases_name / model_type / estimation_column
	out_path = const.EVALUATION_DIR / bases_name / model_type / f"{estimation_column}.csv"
	main(input_csv, estimation_dir, out_path)