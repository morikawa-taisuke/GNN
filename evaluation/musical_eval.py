import numpy as np
import librosa
import pandas as pd
from scipy.stats import kurtosis
from skimage import measure


def analyze_musical_noise(clean_path, enhanced_path, sr=16000):
	# 1. 音声の読み込みとSTFT
	y_clean, _ = librosa.load(clean_path, sr=sr)
	y_est, _ = librosa.load(enhanced_path, sr=sr)

	# 長さを揃える
	min_len = min(len(y_clean), len(y_est))
	y_clean, y_est = y_clean[:min_len], y_est[:min_len]

	S_clean = np.abs(librosa.stft(y_clean, n_fft=512, hop_length=160))
	S_est = np.abs(librosa.stft(y_est, n_fft=512, hop_length=160))

	# 誤差スペクトログラム: |S_clean - S_est|
	error_spec = np.abs(S_clean - S_est)

	# 2. 帯域の定義 (Hz -> index)
	freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
	bands = {
		'Low': (0, 1000),
		'Mid': (1000, 4000),
		'High': (4000, sr // 2)
	}

	results = []

	for band_name, (f_min, f_max) in bands.items():
		# 該当する周波数インデックスを取得
		idx = np.where((freqs >= f_min) & (freqs < f_max))[0]
		if len(idx) == 0: continue

		b_error = error_spec[idx, :]

		# --- 指標1: 尖度 (Kurtosis) ---
		# 誤差が特定の点に集中しているほど高くなる
		kurt_val = kurtosis(b_error.flatten())

		# --- 指標2: 孤立成分の数 (Island Count) ---
		# 誤差の平均より一定以上高い要素を「ノイズの種」とする
		threshold = np.mean(b_error) + 2 * np.std(b_error)
		binary_error = (b_error > threshold).astype(int)
		labels = measure.label(binary_error, connectivity=2)
		island_count = labels.max()  # 孤立した塊の数

		# --- 指標3: 時間的ガタつき (Spectral Flux Variance) ---
		# フレーム間の誤差の変動の激しさを測る
		flux = np.diff(b_error, axis=1) ** 2
		flux_var = np.var(flux)

		# --- 指標4: 平均誤差 (RMSE) ---
		rmse = np.sqrt(np.mean(b_error ** 2))

		results.append({
			'Band': band_name,
			'RMSE': rmse,
			'Kurtosis': kurt_val,
			'IslandCount': island_count,
			'FluxVariance': flux_var
		})

	return results

# 使用例
# data = analyze_musical_noise('clean.wav', 'enhanced.wav')
# df = pd.DataFrame(data)
# df.to_csv('analysis_results.csv', index=False)