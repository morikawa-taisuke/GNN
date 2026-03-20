import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from mymodule import const


def plot_audio_spectrogram(file_path, start_time, end_time, file_name, vmin=-60, vmax=0, cmap='magma', hop_length=64, n_fft = 1024):
	"""
	指定した時間のスペクトログラムをプロットする関数

	Parameters:
	- file_path: 音声ファイルのパス
	- target_time: 抽出したい中心時間（秒）
	- duration: 表示する時間幅（秒）
	- vmin: カラーバーの下限 (dB)
	- vmax: カラーバーの上限 (dB)
	- cmap: カラーマップ (例: 'viridis', 'magma', 'jet', 'inferno')
	- hop_length: フレーム間のサンプリング間隔。小さいほど時間軸が細かくなる（デフォルトは512）
    - n_fft: 窓関数の幅。小さいほど時間分解能が上がるが、周波数分解能は下がる
	"""

	# 1. 指定した区間の音声を読み込み
	duration = end_time - start_time
	if duration <= 0:
		print("エラー: 終了時間は開始時間よりも後に設定してください。")
		return

	y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)

	# 2. STFT（短時間フーリエ変換）の設定
	D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
	S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

	# 3. プロットの作成
	fig, ax = plt.subplots(figsize=(12, 6))

	# スペクトログラムの描画
	# rasterized=True にすることで、PDF出力時に中身が重くなりすぎるのを防ぎつつ、軸はベクターで保持できます
	img = librosa.display.specshow(
		S_db,
		sr=sr,
		hop_length=hop_length,
		x_axis='time',
		y_axis='hz',
		vmin=vmin,
		vmax=vmax,
		cmap=cmap,
		ax=ax,
		x_coords=np.linspace(start_time, end_time, S_db.shape[1]),
		rasterized=True
	)

	# 4. カラーバーの設定
	cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
	cbar.set_label('Intensity (dB)', fontsize=12)

	# 5. ラベルとタイトルの設定（論文用を意識したフォントサイズ）
	ax.set_title(f'{file_name}', fontsize=14)
	ax.set_xlabel('Time (s)', fontsize=12)
	ax.set_ylabel('Frequency (Hz)', fontsize=12)

	# 余白の調整
	plt.tight_layout()

	# 6. 保存と表示
	# PDFで保存（論文用）
	plt.savefig(f"./result/pattern{file_name}.pdf", format='pdf', dpi=300)
	plt.show()


# 実行
if __name__ == "__main__":
	# --- 設定項目 ---
	model_type_list = ["A", "B", "C", "D"]   # "UGAT", "UGCN", "SpeqGAT", "SpeqGCN"
	# select_type_list = ["knn", "random"]
	# area_type_list = ["all", "temporal"]
	wave_type_list = ["reverbe_only"] # "noise_only", "noise_reverb", "reverb_only"
	for model_type in model_type_list:
		# for select_type in select_type_list:
		# 	for area_type in area_type_list:
		for wave_type in wave_type_list:
			# file_name = f"{model_type}_{wave_type}_32node_{area_type}_{select_type}"
			file_name = f"{model_type}"
			# FILE_PATH = f'{const.OUTPUT_WAV_DIR}/Random_Dataset_VCTK_DEMAND_1ch/{model_type}/{file_name}/p232_068_16kHz_hoth_10db.wav'   # p232_068_16kHz_hoth_10db, p232_001_TCAR_01ch_4db_159deg
			# FILE_PATH = f'{const.OUTPUT_WAV_DIR}/Random_Dataset_VCTK_DEMAND_1ch/{model_type}/{file_name}/p232_068_16kHz_hoth_10db_05sec.wav'  # p232_068_16kHz_hoth_10db_05sec.wav, p232_001_TCAR_01ch_4db_0813msec_159deg
			FILE_PATH = f'{const.OUTPUT_WAV_DIR}/Conv-TasNet/All_Model_subset_DEMAND_hoth_1010dB_05sec_4ch_10cm/{model_type}/{wave_type}/p232_068_16kHz_05sec.wav'  # p232_068_16kHz_05sec.wav, p232_001_0813msec_159deg
			START = 0.  # 開始時間（秒）
			END = 1.5  # 終了時間（秒）
			DB_MIN = -70  # カラーバーの下限値
			DB_MAX = 0  # カラーバーの上限値
			COLOR = 'magma'  # カラーマップ（'viridis', 'inferno', 'jet' 等）
			DETAIL = 32  # 時間方向の細かさ（小さいほど高精細。32, 64, 128等）
			hop_length = 64  # 時間分解能を上げるために小さく設定
			n_fft = 1024  # 周波数分解能とのバランスを考慮して設定
			# ファイルが存在する場合のみ実行してください
			try:
				plot_audio_spectrogram(
					file_path=FILE_PATH,
					start_time=START,
					end_time=END,
					vmin=DB_MIN,
					vmax=DB_MAX,
					cmap=COLOR,
					hop_length=DETAIL,
					n_fft=n_fft,
					file_name = file_name
				)
			except Exception as e:
				print(f"エラーが発生しました: {e}")
