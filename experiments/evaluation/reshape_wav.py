import librosa
import soundfile as sf
import os
from pathlib import Path


def process_audio_directory(input_dir, num_segments, output_dir="output_segments"):
	"""
	指定したディレクトリ内のすべての音声ファイルを等分し、0番目を出力する
	"""
	# 1. 出力ディレクトリの作成
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		print(f"ディレクトリ作成: {output_dir}")

	# 2. 対象とする拡張子の定義
	extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

	# 3. フォルダ内のファイルを走査
	input_path = Path(input_dir)
	files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

	if not files:
		print("指定されたディレクトリに音声ファイルが見つかりませんでした。")
		return

	print(f"{len(files)} 個のファイルを処理します...")

	for file_path in files:
		try:
			# 音声の読み込み
			y, sr = librosa.load(file_path, sr=None)

			# 分割計算
			total_samples = len(y)
			segment_length = total_samples // num_segments

			if segment_length == 0:
				print(f"Skipped: {file_path.name} (短すぎます)")
				continue

			# 0番目のセグメント抽出
			segment_0 = y[0: segment_length]

			# 出力ファイル名の生成 (例: original_seg0.wav)
			output_filename = f"{file_path.stem}_seg0.wav"
			save_path = os.path.join(output_dir, output_filename)

			# 保存
			sf.write(save_path, segment_0, sr)
			print(f"Done: {file_path.name} -> {output_filename}")

		except Exception as e:
			print(f"Error: {file_path.name} の処理中にエラーが発生しました: {e}")


# --- 設定 ---
TARGET_DIR = "./my_audio_folder"  # 音声ファイルが入っているフォルダパス
SEGMENT_COUNT = 2  # 何分割するか
OUT_DIR = "./processed_results"  # 保存先フォルダ

# 実行
process_audio_directory(TARGET_DIR, SEGMENT_COUNT, OUT_DIR)