import sys
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import torch
import ast  # 斁E���E化されたリストを読み込むために忁E��E


class CsvDataset(Dataset):
	"""
	CSVファイルからファイルパスを読み込み、E��声チE�EタをロードするDatasetクラス、E

	Args:
		csv_path (str): チE�EタセチE��のパス惁E��が記載されたCSVファイルのパス、E
		input_column_header (str): 入力データとして使用するCSVの列名
								   (侁E 'noise_only_path', 'noise_reverb_path')、E
		chunk_size (int): 音声チE�Eタを�E割する際�Eチャンクサイズ�E�サンプル数�E�、E
		sample_rate (int): 音声チE�Eタのサンプリングレート！Ez�E�、E
		max_length_sec (int): 音声チE�Eタの最大長�E�秒）。これを趁E��る場合�E刁E��捨てる、E
	"""

	def __init__(self, csv_path, input_column_header, chunk_size=16384 * 4, sample_rate=16000, max_length_sec=None):

		super(CsvDataset, self).__init__()

		self.chunk_size = chunk_size
		self.teacher_column = "clean"  # 教師チE�Eタは常に 'clean_path' を使用
		self.input_column = input_column_header
		if max_length_sec is not None:
			self.max_length_samples = max_length_sec * sample_rate
		else:
			self.max_length_samples = None

		# --- CSVファイルの読み込み ---
		try:
			self.data_df = pd.read_csv(csv_path)
		except FileNotFoundError:
			print(f"❁Eエラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列�E存在確誁E---
		if self.teacher_column not in self.data_df.columns:
			print(f"❁Eエラー: CSVに教師チE�Eタ用の刁E'{self.teacher_column}' が見つかりません、E, file=sys.stderr)
			sys.exit(1)
		if self.input_column not in self.data_df.columns:
			print(f"❁Eエラー: CSVに入力データ用の刁E'{self.input_column}' が見つかりません、E, file=sys.stderr)
			sys.exit(1)

		# --- 欠損値�E�空のパス�E�を持つ行を削除 ---
		original_len = len(self.data_df)
		self.data_df.dropna(subset=[self.teacher_column, self.input_column], inplace=True)
		self.data_df = self.data_df[(self.data_df[self.teacher_column] != "") & (self.data_df[self.input_column] != "")]

		if len(self.data_df) < original_len:
			print(f"⚠�E�E 注愁E {original_len - len(self.data_df)}行�EチE�Eタパスに欠損があったため、E��外されました、E)

		print(f"✁E{csv_path} から {len(self.data_df)} 件のファイルペアを読み込みました、E)
		print(f"  - 入力データ: '{self.input_column}' 列を使用")
		print(f"  - 教師チE�Eタ: '{self.teacher_column}' 列を使用")

	def __getitem__(self, index):
		"""
		持E��されたインチE��クスのチE�Eタ�E��E力と教師�E�をロードし、E��さを調整して返す、E
		Returns:
			noisy_waveform (torch.Tensor): 入力音声波形 [Channels=1, TimeSteps]
			clean_waveform (torch.Tensor): 教師音声波形 [Channels=1, TimeSteps]
		"""
		# --- 1. ファイルパスの取征E---
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
		チE�EタセチE��の総数を返す、E
		"""
		return len(self.data_df)

	@staticmethod
	def collate_fn(batch):
		"""バッチ�EのチE��ソルサイズを揃えるためのカスタムcollate関数"""
		# バッチ�Eの最大長を見つける
		max_len = max([x[0].size(-1) for x in batch])

		# 全てのチE��ソルを最大長にパディング
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
 ▼▼▼ [改良牁E 推論用チE�Eタローダ ▼▼▼
===================================================================
"""
class CsvInferenceDataset(Dataset):
	"""
	推論用に、CSVファイルから入力音声のファイルパスを読み込むDatasetクラス、E

	Args:
		csv_path (str): チE�EタセチE��のパス惁E��が記載されたCSVファイルのパス、E
		input_column_header (str): 入力データとして使用するCSVの列名、E
		sample_rate (int): 音声チE�Eタのサンプリングレート！Ez�E�、E
	"""

	def __init__(self, csv_path, input_column_header, sample_rate=16000):
		super(CsvInferenceDataset, self).__init__()

		self.input_column = input_column_header
		self.sample_rate = sample_rate

		# --- CSVファイルの読み込み ---
		try:
			self.data_df = pd.read_csv(csv_path)
		except FileNotFoundError:
			print(f"❁Eエラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列�E存在確誁E---
		if self.input_column not in self.data_df.columns:
			print(f"❁Eエラー: CSVに入力データ用の刁E'{self.input_column}' が見つかりません、E, file=sys.stderr)
			sys.exit(1)

		# --- 欠損値�E�空のパス�E�を持つ行を削除 ---
		original_len = len(self.data_df)
		self.data_df.dropna(subset=[self.input_column], inplace=True)
		self.data_df = self.data_df[self.data_df[self.input_column] != ""]
		if len(self.data_df) < original_len:
			print(f"⚠�E�E 注愁E {original_len - len(self.data_df)}行�EチE�Eタパスに欠損があったため、E��外されました、E)

		print(f"✁E{csv_path} から {len(self.data_df)} 件の音声ファイルを読み込みました、E)
		print(f"  - 入力データ: '{self.input_column}' 列を使用")

	def __getitem__(self, index):
		"""
		持E��されたインチE��クスのチE�Eタをロードし、波形とファイル名を返す、E
		"""
		# --- 1. ファイルパスの取征E---
		row = self.data_df.iloc[index]
		noisy_path = row[self.input_column]
		# print("noisy_path:", noisy_path)

		# --- 2. 音声の読み込み ---
		noisy_waveform, current_sample_rate = torchaudio.load(noisy_path)

		# --- 3. リサンプリング�E�忁E��に応じて�E�E---
		if current_sample_rate != self.sample_rate:
			resampler = torchaudio.transforms.Resample(current_sample_rate, self.sample_rate)
			noisy_waveform = resampler(noisy_waveform)

		# --- 4. ファイル名�E取得（拡張子なし！E---
		file_name = os.path.splitext(os.path.basename(noisy_path))[0]

		return noisy_waveform, file_name

	def __len__(self):
		"""
		チE�EタセチE��の総数を返す、E
		"""
		return len(self.data_df)


"""
CSVファイルからファイルパスと褁E��の残響特徴量を読み込むDatasetクラス、E
"""
class ReverbEncoderDataset(Dataset):
	def __init__(self, csv_path, input_column_header,
	             # 読み込む補助特徴量カラムのリスチE(CSVの列名と一致させめE
	             reverb_feature_columns=["cepstrum_coeffs", "rt60", "c50", "d50"],
	             chunk_size=16384 * 4, sample_rate=16000, max_length_sec=None):

		super(ReverbEncoderDataset, self).__init__()

		self.chunk_size = chunk_size
		self.teacher_column = "clean_path"  # 教師チE�Eタは 'clean_path' を使用
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
			print(f"❁Eエラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列�E存在確誁E---
		required_cols = [self.teacher_column, self.input_column] + self.reverb_feature_columns

		for col in required_cols:
			if col not in self.data_df.columns:
				# CSVのヘッダーぁE'clean_path' ではなぁE'clean' の可能性があるため調整 (允E�EDataset/generate_reverb_dataset.pyでは'clean_path'だが、ここではフォールバックの柔軟性を老E�E)
				if col == "clean_path" and "clean" in self.data_df.columns:
					self.teacher_column = "clean"
					continue

				# 残響特徴量�E列がなかった場合、エラーで停止
				print(f"❁Eエラー: CSVに忁E��な刁E'{col}' が見つかりません、E)
				sys.exit(1)

		# --- 欠損値�E�空のパス�E�を持つ行を削除 ---
		original_len = len(self.data_df)
		subset_cols = [self.teacher_column, self.input_column] + self.reverb_feature_columns
		self.data_df.dropna(subset=subset_cols, inplace=True)
		self.data_df = self.data_df[(self.data_df[self.teacher_column] != "") & (self.data_df[self.input_column] != "")]

		if len(self.data_df) < original_len:
			print(f"⚠�E�E 注愁E {original_len - len(self.data_df)}行�EチE�Eタパスに欠損があったため、E��外されました、E)

		print(f"✁E{csv_path} から {len(self.data_df)} 件のファイルペアを読み込みました、E)
		print(f"  - 入力データ: '{self.input_column}' 刁E 教師音声: '{self.teacher_column}' 列を使用")
		print(f"  - 教師残響特徴釁E {self.reverb_feature_columns} を連結して使用")

	def __getitem__(self, index):
		"""
		持E��されたインチE��クスのチE�Eタ�E��E力、教師、教師残響特徴量）をロードして返す、E
		"""
		# --- 1. ファイルパスの取征E---
		row = self.data_df.iloc[index]
		clean_path = Path(row[self.teacher_column])
		noisy_path = Path(row[self.input_column])

		clean_waveform, current_sample_rate = torchaudio.load(clean_path)
		noisy_waveform, _ = torchaudio.load(noisy_path)

		# サンプリングレート�E確認とリサンプリング�E�忁E��に応じて�E�E
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

		# --- 3. 教師残響特徴量�Eロードと連絁E---
		feature_tensors = []
		for col in self.reverb_feature_columns:
			feature_value = row[col]

			try:
				if col == "cepstrum_coeffs":
					# 斁E���E化されたリストを ast.literal_eval でPythonリストに変換
					feature_list = ast.literal_eval(feature_value)
					feature_np = np.array(feature_list, dtype=np.float32)
					feature_tensor = torch.from_numpy(feature_np)  # 侁E [16]次允E�Eクトル
				else:
					# RT60, C50, D50などのスカラー値をテンソル匁E
					feature_tensor = torch.tensor([float(feature_value)], dtype=torch.float32)  # 侁E [1]次允E��カラー
			except Exception as e:
				# チE�Eタの破損や形式エラーに対忁E
				# cepstrum_coeffs�E�EPC=16�E��E16次允E��他�Eスカラー�E�E次允E��と仮宁E
				dim = 16 if col == "cepstrum_coeffs" else 1
				feature_tensor = torch.zeros(dim, dtype=torch.float32)

			feature_tensors.append(feature_tensor)

		# 全ての教師特徴量を連絁E(侁E [16] + [1] + [1] + [1] -> [19])
		reverb_feature_tensor = torch.cat(feature_tensors, dim=0)

		# ☁E��更: 教師残響特徴釁E(reverb_feature_tensor) をリターンに追加
		return noisy_waveform, clean_waveform, reverb_feature_tensor

	def __len__(self):
		return len(self.data_df)

	@staticmethod
	def collate_fn(batch):
		"""バッチ�EのチE��ソルサイズを揃えるためのカスタムcollate関数"""
		# バッチ�Eの最大長を見つける
		max_len = max([x[0].size(-1) for x in batch])

		# 全てのチE��ソルを最大長にパディング
		padded_batch = []
		reverb_features = []

		# ☁E��更: バッチかめEつの要素 (波形2つと特徴釁Eつ) をアンパック
		for mix_data, target_data, reverb_feature_tensor in batch:
			pad_mix = F.pad(mix_data, (0, max_len - mix_data.size(-1)))
			pad_target = F.pad(target_data, (0, max_len - target_data.size(-1)))
			padded_batch.append((pad_mix, pad_target))
			reverb_features.append(reverb_feature_tensor)

		# 波形チE�Eタのバッチ化
		mix_data = torch.stack([x[0] for x in padded_batch])
		target_data = torch.stack([x[1] for x in padded_batch])

		# 特徴量テンソルのバッチ化
		reverb_features_batch = torch.stack(reverb_features)

		# ☁E��更: 3つのバッチ化されたテンソルを返す
		return mix_data, target_data, reverb_features_batch




# ===================================================================
# ▼▼▼ 使ぁE���E�サンプルコード！E▼▼▼
# ===================================================================
if __name__ == "__main__":
	# --- こ�Eスクリプトを直接実行した際のチE��トコーチE---

	# 1. チE��ト用のCSVファイルを作�E (実際には既存�ECSVを使ぁE
	print("--- チE��ト用のCSVファイルを作�EしてぁE��ぁE---")
	dummy_csv_path = "test_data.csv"
	dummy_data = {
		"clean": ["clean_a.wav", "clean_b.wav", "clean_c.wav"],
		"noise_only": ["noise_a.wav", "noise_b.wav", ""],  # cは欠搁E
		"noise_reverb": ["noise_reverb_a.wav", "noise_reverb_b.wav", "noise_reverb_c.wav"],
	}
	pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)

	# ダミ�Eの音声ファイルを作�E
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
	print("--- チE��トファイルの準備完亁E---\n")

	# 2. チE�EタセチE��のインスタンスを作�E
	print("--- チE�EタセチE��のインスタンスを作�EしまぁE---")
	# 入力として「雑音�E�残響」�E列を持E��E
	input_header = "noise_reverb"
	train_dataset = CsvDataset(csv_path=dummy_csv_path, input_column_header=input_header)

	# 3. DataLoaderを作�E
	from torch.utils.data import DataLoader

	train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

	# 4. チE�EタめEバッチ取り�Eして形状を確誁E
	print("\n--- DataLoaderからチE�Eタを取り�EしまぁE---")
	# `iter`でイチE��ータを作�Eし、`next`で最初�Eバッチを取征E
	noisy_signal, clean_signal = next(iter(train_loader))

	print(f"取得したデータの形状:")
	print(f"  - 入力信号 (Noisy signal): {noisy_signal.shape}")
	print(f"  - 教師信号 (Clean signal): {clean_signal.shape}")

	# 形状の解説: (バッチサイズ, 周波数ビン数, 時間フレーム数)
	# 周波数ビン数 = n_fft / 2 + 1 = 512 / 2 + 1 = 257
	# 時間フレーム数 = chunk_size / hop_length = (16384 * 4) / 128 = 512

	# --- 入力�Eを変更してチE��チE---
	print("\n--- 入力�Eを変更して再度チE��トしまぁE---")
	input_header_2 = "noise_only"
	train_dataset_2 = CsvDataset(csv_path=dummy_csv_path, input_column_header=input_header_2)
	# noise_c.wavが欠損してぁE��ため、データ数は2件になる�EぁE
	assert len(train_dataset_2) == 2, "欠損データが正しく除外されてぁE��せん"
	print("✁E欠損データの除外を正しく確認しました、E)
