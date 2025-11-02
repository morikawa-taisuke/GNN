import sys
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import torch
import ast  # 文字列化されたリストを読み込むために必要


class CsvDataset(Dataset):
	"""
	CSVファイルからファイルパスを読み込み、音声データをロードするDatasetクラス。

	Args:
		csv_path (str): データセットのパス情報が記載されたCSVファイルのパス。
		input_column_header (str): 入力データとして使用するCSVの列名
								   (例: 'noise_only_path', 'noise_reverb_path')。
		chunk_size (int): 音声データを分割する際のチャンクサイズ（サンプル数）。
		sample_rate (int): 音声データのサンプリングレート（Hz）。
		max_length_sec (int): 音声データの最大長（秒）。これを超える場合は切り捨てる。
	"""

	def __init__(self, csv_path, input_column_header, chunk_size=16384 * 4, sample_rate=16000, max_length_sec=None):

		super(CsvDataset, self).__init__()

		self.chunk_size = chunk_size
		self.teacher_column = "clean"  # 教師データは常に 'clean_path' を使用
		self.input_column = input_column_header
		if max_length_sec is not None:
			self.max_length_samples = max_length_sec * sample_rate
		else:
			self.max_length_samples = None

		# --- CSVファイルの読み込み ---
		try:
			self.data_df = pd.read_csv(csv_path)
		except FileNotFoundError:
			print(f"❌ エラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列の存在確認 ---
		if self.teacher_column not in self.data_df.columns:
			print(f"❌ エラー: CSVに教師データ用の列 '{self.teacher_column}' が見つかりません。", file=sys.stderr)
			sys.exit(1)
		if self.input_column not in self.data_df.columns:
			print(f"❌ エラー: CSVに入力データ用の列 '{self.input_column}' が見つかりません。", file=sys.stderr)
			sys.exit(1)

		# --- 欠損値（空のパス）を持つ行を削除 ---
		original_len = len(self.data_df)
		self.data_df.dropna(subset=[self.teacher_column, self.input_column], inplace=True)
		self.data_df = self.data_df[(self.data_df[self.teacher_column] != "") & (self.data_df[self.input_column] != "")]

		if len(self.data_df) < original_len:
			print(f"⚠️  注意: {original_len - len(self.data_df)}行のデータパスに欠損があったため、除外されました。")

		print(f"✅ {csv_path} から {len(self.data_df)} 件のファイルペアを読み込みました。")
		print(f"  - 入力データ: '{self.input_column}' 列を使用")
		print(f"  - 教師データ: '{self.teacher_column}' 列を使用")

	def __getitem__(self, index):
		"""
		指定されたインデックスのデータ（入力と教師）をロードし、STFTを適用して返す。
		"""
		# --- 1. ファイルパスの取得 ---
		row = self.data_df.iloc[index]
		clean_path = Path(row[self.teacher_column])
		noisy_path = Path(row[self.input_column])

		clean_waveform, current_sample_rate = torchaudio.load(clean_path)
		noisy_waveform, _ = torchaudio.load(noisy_path)

		if self.max_length_samples is not None:
			if noisy_waveform.shape[-1] > self.max_length_samples:
				noisy_waveform = noisy_waveform[:, :self.max_length_samples]
				clean_waveform = clean_waveform[:, :self.max_length_samples]
			# elif noisy_waveform.shape[-1] < self.max_length_samples:
			# 	padding_amount = self.max_length_samples - noisy_waveform.shape[1]
			# 	noisy_waveform = F.pad(noisy_waveform, (0, padding_amount))
			# 	clean_waveform = F.pad(clean_waveform, (0, padding_amount))

		return noisy_waveform, clean_waveform

	def __len__(self):
		"""
		データセットの総数を返す。
		"""
		return len(self.data_df)

	@staticmethod
	def collate_fn(batch):
		"""バッチ内のテンソルサイズを揃えるためのカスタムcollate関数"""
		# バッチ内の最大長を見つける
		max_len = max([x[0].size(-1) for x in batch])

		# 全てのテンソルを最大長にパディング
		padded_batch = []
		for mix_data, target_data in batch:
			pad_mix = F.pad(mix_data, (0, max_len - mix_data.size(-1)))
			pad_target = F.pad(target_data, (0, max_len - target_data.size(-1)))
			padded_batch.append((pad_mix, pad_target))

		# バッチ化
		mix_data = torch.stack([x[0] for x in padded_batch])
		target_data = torch.stack([x[1] for x in padded_batch])

		return mix_data, target_data


""" 
===================================================================
 ▼▼▼ [改良版] 推論用データローダ ▼▼▼
===================================================================
"""
class CsvInferenceDataset(Dataset):
	"""
	推論用に、CSVファイルから入力音声のファイルパスを読み込むDatasetクラス。

	Args:
		csv_path (str): データセットのパス情報が記載されたCSVファイルのパス。
		input_column_header (str): 入力データとして使用するCSVの列名。
		sample_rate (int): 音声データのサンプリングレート（Hz）。
	"""

	def __init__(self, csv_path, input_column_header, sample_rate=16000):
		super(CsvInferenceDataset, self).__init__()

		self.input_column = input_column_header
		self.sample_rate = sample_rate

		# --- CSVファイルの読み込み ---
		try:
			self.data_df = pd.read_csv(csv_path)
		except FileNotFoundError:
			print(f"❌ エラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列の存在確認 ---
		if self.input_column not in self.data_df.columns:
			print(f"❌ エラー: CSVに入力データ用の列 '{self.input_column}' が見つかりません。", file=sys.stderr)
			sys.exit(1)

		# --- 欠損値（空のパス）を持つ行を削除 ---
		original_len = len(self.data_df)
		self.data_df.dropna(subset=[self.input_column], inplace=True)
		self.data_df = self.data_df[self.data_df[self.input_column] != ""]
		if len(self.data_df) < original_len:
			print(f"⚠️  注意: {original_len - len(self.data_df)}行のデータパスに欠損があったため、除外されました。")

		print(f"✅ {csv_path} から {len(self.data_df)} 件の音声ファイルを読み込みました。")
		print(f"  - 入力データ: '{self.input_column}' 列を使用")

	def __getitem__(self, index):
		"""
		指定されたインデックスのデータをロードし、波形とファイル名を返す。
		"""
		# --- 1. ファイルパスの取得 ---
		row = self.data_df.iloc[index]
		noisy_path = row[self.input_column]
		# print("noisy_path:", noisy_path)

		# --- 2. 音声の読み込み ---
		noisy_waveform, current_sample_rate = torchaudio.load(noisy_path)

		# --- 3. リサンプリング（必要に応じて） ---
		if current_sample_rate != self.sample_rate:
			resampler = torchaudio.transforms.Resample(current_sample_rate, self.sample_rate)
			noisy_waveform = resampler(noisy_waveform)

		# --- 4. ファイル名の取得（拡張子なし） ---
		file_name = os.path.splitext(os.path.basename(noisy_path))[0]

		return noisy_waveform, file_name

	def __len__(self):
		"""
		データセットの総数を返す。
		"""
		return len(self.data_df)


"""
CSVファイルからファイルパスと複数の残響特徴量を読み込むDatasetクラス。
"""
class ReverbEncoderDataset(Dataset):
	def __init__(self, csv_path, input_column_header,
	             # 読み込む補助特徴量カラムのリスト (CSVの列名と一致させる)
	             reverb_feature_columns=["cepstrum_coeffs", "rt60", "c50", "d50"],
	             chunk_size=16384 * 4, sample_rate=16000, max_length_sec=None):

		super(ReverbEncoderDataset, self).__init__()

		self.chunk_size = chunk_size
		self.teacher_column = "clean_path"  # 教師データは 'clean_path' を使用
		self.input_column = input_column_header
		self.reverb_feature_columns = reverb_feature_columns
		self.sample_rate = sample_rate

		if max_length_sec is not None:
			self.max_length_samples = max_length_sec * sample_rate
		else:
			self.max_length_samples = None

		# --- CSVファイルの読み込み ---
		try:
			self.data_df = pd.read_csv(csv_path)
		except FileNotFoundError:
			print(f"❌ エラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列の存在確認 ---
		required_cols = [self.teacher_column, self.input_column] + self.reverb_feature_columns

		for col in required_cols:
			if col not in self.data_df.columns:
				# CSVのヘッダーが 'clean_path' ではなく 'clean' の可能性があるため調整 (元のDataset/generate_reveb_dataset.pyでは'clean_path'だが、ここではフォールバックの柔軟性を考慮)
				if col == "clean_path" and "clean" in self.data_df.columns:
					self.teacher_column = "clean"
					continue

				# 残響特徴量の列がなかった場合、エラーで停止
				print(f"❌ エラー: CSVに必要な列 '{col}' が見つかりません。")
				sys.exit(1)

		# --- 欠損値（空のパス）を持つ行を削除 ---
		original_len = len(self.data_df)
		subset_cols = [self.teacher_column, self.input_column] + self.reverb_feature_columns
		self.data_df.dropna(subset=subset_cols, inplace=True)
		self.data_df = self.data_df[(self.data_df[self.teacher_column] != "") & (self.data_df[self.input_column] != "")]

		if len(self.data_df) < original_len:
			print(f"⚠️  注意: {original_len - len(self.data_df)}行のデータパスに欠損があったため、除外されました。")

		print(f"✅ {csv_path} から {len(self.data_df)} 件のファイルペアを読み込みました。")
		print(f"  - 入力データ: '{self.input_column}' 列, 教師音声: '{self.teacher_column}' 列を使用")
		print(f"  - 教師残響特徴量: {self.reverb_feature_columns} を連結して使用")

	def __getitem__(self, index):
		"""
		指定されたインデックスのデータ（入力、教師、教師残響特徴量）をロードして返す。
		"""
		# --- 1. ファイルパスの取得 ---
		row = self.data_df.iloc[index]
		clean_path = Path(row[self.teacher_column])
		noisy_path = Path(row[self.input_column])

		clean_waveform, current_sample_rate = torchaudio.load(clean_path)
		noisy_waveform, _ = torchaudio.load(noisy_path)

		# サンプリングレートの確認とリサンプリング（必要に応じて）
		if current_sample_rate != self.sample_rate:
			resampler = torchaudio.transforms.Resample(current_sample_rate, self.sample_rate)
			clean_waveform = resampler(clean_waveform)
			noisy_waveform = resampler(noisy_waveform)

		# --- 2. 音声波形のロードと長さ調整 ---
		if self.max_length_samples is not None:
			if noisy_waveform.shape[-1] > self.max_length_samples:
				noisy_waveform = noisy_waveform[:, :self.max_length_samples]
				clean_waveform = clean_waveform[:, :self.max_length_samples]
			elif noisy_waveform.shape[-1] < self.max_length_samples:
				padding_amount = self.max_length_samples - noisy_waveform.shape[1]
				noisy_waveform = F.pad(noisy_waveform, (0, padding_amount))
				clean_waveform = F.pad(clean_waveform, (0, padding_amount))

		# --- 3. 教師残響特徴量のロードと連結 ---
		feature_tensors = []
		for col in self.reverb_feature_columns:
			feature_value = row[col]

			try:
				if col == "cepstrum_coeffs":
					# 文字列化されたリストを ast.literal_eval でPythonリストに変換
					feature_list = ast.literal_eval(feature_value)
					feature_np = np.array(feature_list, dtype=np.float32)
					feature_tensor = torch.from_numpy(feature_np)  # 例: [16]次元ベクトル
				else:
					# RT60, C50, D50などのスカラー値をテンソル化
					feature_tensor = torch.tensor([float(feature_value)], dtype=torch.float32)  # 例: [1]次元スカラー
			except Exception as e:
				# データの破損や形式エラーに対応
				# cepstrum_coeffs（LPC=16）は16次元、他はスカラー（1次元）と仮定
				dim = 16 if col == "cepstrum_coeffs" else 1
				feature_tensor = torch.zeros(dim, dtype=torch.float32)

			feature_tensors.append(feature_tensor)

		# 全ての教師特徴量を連結 (例: [16] + [1] + [1] + [1] -> [19])
		reverb_feature_tensor = torch.cat(feature_tensors, dim=0)

		# ★変更: 教師残響特徴量 (reverb_feature_tensor) をリターンに追加
		return noisy_waveform, clean_waveform, reverb_feature_tensor

	def __len__(self):
		return len(self.data_df)

	@staticmethod
	def collate_fn(batch):
		"""バッチ内のテンソルサイズを揃えるためのカスタムcollate関数"""
		# バッチ内の最大長を見つける
		max_len = max([x[0].size(-1) for x in batch])

		# 全てのテンソルを最大長にパディング
		padded_batch = []
		reverb_features = []

		# ★変更: バッチから3つの要素 (波形2つと特徴量1つ) をアンパック
		for mix_data, target_data, reverb_feature_tensor in batch:
			pad_mix = F.pad(mix_data, (0, max_len - mix_data.size(-1)))
			pad_target = F.pad(target_data, (0, max_len - target_data.size(-1)))
			padded_batch.append((pad_mix, pad_target))
			reverb_features.append(reverb_feature_tensor)

		# 波形データのバッチ化
		mix_data = torch.stack([x[0] for x in padded_batch])
		target_data = torch.stack([x[1] for x in padded_batch])

		# 特徴量テンソルのバッチ化
		reverb_features_batch = torch.stack(reverb_features)

		# ★変更: 3つのバッチ化されたテンソルを返す
		return mix_data, target_data, reverb_features_batch




# ===================================================================
# ▼▼▼ 使い方（サンプルコード） ▼▼▼
# ===================================================================
if __name__ == "__main__":
	# --- このスクリプトを直接実行した際のテストコード ---

	# 1. テスト用のCSVファイルを作成 (実際には既存のCSVを使う)
	print("--- テスト用のCSVファイルを作成しています ---")
	dummy_csv_path = "test_data.csv"
	dummy_data = {
		"clean": ["clean_a.wav", "clean_b.wav", "clean_c.wav"],
		"noise_only": ["noise_a.wav", "noise_b.wav", ""],  # cは欠損
		"noise_reverbe": ["noise_reverb_a.wav", "noise_reverb_b.wav", "noise_reverb_c.wav"],
	}
	pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)

	# ダミーの音声ファイルを作成
	import soundfile as sf

	for name in [
		"clean_a",
		"clean_b",
		"clean_c",
		"noise_a",
		"noise_b",
		"noise_reverb_a",
		"noise_reverb_b",
		"noise_reverb_c",
	]:
		sf.write(f"{name}.wav", np.random.randn(16384 * 5), 16000)
	print("--- テストファイルの準備完了 ---\n")

	# 2. データセットのインスタンスを作成
	print("--- データセットのインスタンスを作成します ---")
	# 入力として「雑音＋残響」の列を指定
	input_header = "noise_reverbe"
	train_dataset = CsvDataset(csv_path=dummy_csv_path, input_column_header=input_header)

	# 3. DataLoaderを作成
	from torch.utils.data import DataLoader

	train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

	# 4. データを1バッチ取り出して形状を確認
	print("\n--- DataLoaderからデータを取り出します ---")
	# `iter`でイテレータを作成し、`next`で最初のバッチを取得
	noisy_signal, clean_signal = next(iter(train_loader))

	print(f"取得したデータの形状:")
	print(f"  - 入力信号 (Noisy signal): {noisy_signal.shape}")
	print(f"  - 教師信号 (Clean signal): {clean_signal.shape}")

	# 形状の解説: (バッチサイズ, 周波数ビン数, 時間フレーム数)
	# 周波数ビン数 = n_fft / 2 + 1 = 512 / 2 + 1 = 257
	# 時間フレーム数 = chunk_size / hop_length = (16384 * 4) / 128 = 512

	# --- 入力列を変更してテスト ---
	print("\n--- 入力列を変更して再度テストします ---")
	input_header_2 = "noise_only"
	train_dataset_2 = CsvDataset(csv_path=dummy_csv_path, input_column_header=input_header_2)
	# noise_c.wavが欠損しているため、データ数は2件になるはず
	assert len(train_dataset_2) == 2, "欠損データが正しく除外されていません"
	print("✅ 欠損データの除外を正しく確認しました。")
