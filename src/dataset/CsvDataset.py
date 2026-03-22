import sys
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import torch
import ast  # 文字列化されたリストを読み込むために忁EE


class CsvDataset(Dataset):
	"""
	CSVファイルからファイルパスを読み込み、音声データをロードする学習・検証用Datasetクラス。

	指定された入力波形と教師波形（cleanなど）のペアを読み込み、必要に応じて長さを切り詰めて返します。

	Args:
		csv_path (str): データセットのパス情報が記載されたCSVファイルのパス
		input_column_header (str): 入力データとして使用するCSVの列名 (例: 'noise_only_path', 'noise_reverb_path')
		chunk_size (int): 音声データを切り出す・読み込む基本チャンクサイズ(サンプル数)
		sample_rate (int): 音声データのサンプリングレート
		max_length_sec (float, optional): 音声データの最大長(秒)。これを超える場合は先頭から切り詰められます
	"""

	def __init__(self, csv_path, input_column_header, chunk_size=16384 * 4, sample_rate=16000, max_length_sec=None):

		super(CsvDataset, self).__init__()

		self.chunk_size = chunk_size
		self.teacher_column = "clean"  # 教師チEEタは常に 'clean_path' を使用
		self.input_column = input_column_header
		if max_length_sec is not None:
			self.max_length_samples = max_length_sec * sample_rate
		else:
			self.max_length_samples = None

		# --- CSVファイルの読み込み ---
		try:
			self.data_df = pd.read_csv(csv_path)
		except FileNotFoundError:
			print(f"エラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列の存在確認---
		if self.teacher_column not in self.data_df.columns:
			print(f"エラー: CSVに教師チEEタ用の列,'{self.teacher_column}' が見つかりません。", file=sys.stderr)
			sys.exit(1)
		if self.input_column not in self.data_df.columns:
			print(f"エラー: CSVに入力データ用の列,'{self.input_column}' が見つかりません。", file=sys.stderr)
			sys.exit(1)

		# --- 欠損値（空のパス）を持つ行を削除 ---
		original_len = len(self.data_df)
		self.data_df.dropna(subset=[self.teacher_column, self.input_column], inplace=True)
		self.data_df = self.data_df[(self.data_df[self.teacher_column] != "") & (self.data_df[self.input_column] != "")]

		if len(self.data_df) < original_len:
			print(f"注意: {original_len - len(self.data_df)}行のデータパスに欠損があったため、除外されました。")

		print(f"{csv_path} から {len(self.data_df)} 件のファイルペアを読み込みました。")
		print(f"  - 入力データ: '{self.input_column}' 列を使用")
		print(f"  - 教師チEEタ: '{self.teacher_column}' 列を使用")

	def __getitem__(self, index):
		"""
		指定されたインデックスのデータをロードし、入力と教師のペアを返す。

		長さを制限する設定（max_length_samples）がある場合は、末尾を切り取って調整します。

		Args:
			index (int): 取得対象のデータインデックス

		Returns:
			tuple[torch.Tensor, torch.Tensor]:
				- noisy_waveform (torch.Tensor): 入力音声波形 [Channels=1, TimeSteps]
				- clean_waveform (torch.Tensor): 教師音声波形 [Channels=1, TimeSteps]
		"""
		# --- 1. ファイルパスの取得---
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
		データセットの総数を返す、E
		"""
		return len(self.data_df)

	@staticmethod
	def collate_fn(batch):
		"""
		バッチ内のテンソルサイズを揃えるためのカスタムcollate関数。

		DataLoaderでミニバッチを作成する際、データの時間長が異なる場合に
		バッチ内の最大長を見つけて各データをゼロパディングします。

		Args:
			batch (list): __getitem__ で取得された要素（タプル）のリスト

		Returns:
			tuple[torch.Tensor, torch.Tensor]: バッチ化されたパディング済み入力と教師テンソル群
		"""
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
 ▼▼▼ [改良牁E 推論用チEEタローダ ▼▼▼
===================================================================
"""
class CsvInferenceDataset(Dataset):
	"""
	推論（テスト）用に、CSVファイルから入力音声のみを読み込むDatasetクラス。

	教師データが存在しない本番環境やテストデータでの分離推論を行うために使用されます。
	また、推論結果を保存するためのファイル名も合わせて返却します。

	Args:
		csv_path (str): データセットのパス情報が記載されたCSVファイルのパス
		input_column_header (str): 入力データとして使用するCSVの列名
		sample_rate (int): 期待するサンプリングレート（一致しない場合は自動リサンプリング）
	"""

	def __init__(self, csv_path, input_column_header, sample_rate=16000):
		super(CsvInferenceDataset, self).__init__()

		self.input_column = input_column_header
		self.sample_rate = sample_rate

		# --- CSVファイルの読み込み ---
		try:
			self.data_df = pd.read_csv(csv_path)
		except FileNotFoundError:
			print(f"エラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列の存在確認---
		if self.input_column not in self.data_df.columns:
			print(f"エラー: CSVに入力データ用の列,'{self.input_column}' が見つかりません。", file=sys.stderr)
			sys.exit(1)

		# --- 欠損値（空のパス）を持つ行を削除 ---
		original_len = len(self.data_df)
		self.data_df.dropna(subset=[self.input_column], inplace=True)
		self.data_df = self.data_df[self.data_df[self.input_column] != ""]
		if len(self.data_df) < original_len:
			print(f"注意: {original_len - len(self.data_df)}行のデータパスに欠損があったため、除外されました。")

		print(f"{csv_path} から {len(self.data_df)} 件の音声ファイルを読み込みました。")
		print(f"  - 入力データ: '{self.input_column}' 列を使用")

	def __getitem__(self, index):
		"""
		指定されたインデックスのデータをロードし、波形と元ファイル名を返す。

		Args:
			index (int): 取得対象のデータインデックス

		Returns:
			tuple[torch.Tensor, str]:
				- noisy_waveform (torch.Tensor): 入力音声波形
				- file_name (str): 拡張子を除いたベースファイル名（出力保存時に使用）
		"""
		# --- 1. ファイルパスの取得---
		row = self.data_df.iloc[index]
		noisy_path = row[self.input_column]
		# print("noisy_path:", noisy_path)

		# --- 2. 音声の読み込み ---
		noisy_waveform, current_sample_rate = torchaudio.load(noisy_path)

		# --- 3. リサンプリングE忁Eに応じてEE---
		if current_sample_rate != self.sample_rate:
			resampler = torchaudio.transforms.Resample(current_sample_rate, self.sample_rate)
			noisy_waveform = resampler(noisy_waveform)

		# --- 4. ファイル名E取得（拡張子なし！---
		file_name = os.path.splitext(os.path.basename(noisy_path))[0]

		return noisy_waveform, file_name

	def __len__(self):
		"""
		データセットの総数を返す、E
		"""
		return len(self.data_df)


"""
CSVファイルからファイルパスと残響の物理特徴量（RT60など）を読み込むDatasetクラス
"""
class ReverbEncoderDataset(Dataset):
	def __init__(self, csv_path, input_column_header,
	             # 読み込む補助特徴量カラムのリスチE(CSVの列名と一致させめE
	             reverb_feature_columns=["cepstrum_coeffs", "rt60", "c50", "d50"],
	             chunk_size=16384 * 4, sample_rate=16000, max_length_sec=None):

		super(ReverbEncoderDataset, self).__init__()

		self.chunk_size = chunk_size
		self.teacher_column = "clean_path"  # 教師チEEタは 'clean_path' を使用
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
			print(f"エラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
			sys.exit(1)

		# --- 列の存在確認---
		required_cols = [self.teacher_column, self.input_column] + self.reverb_feature_columns

		for col in required_cols:
			if col not in self.data_df.columns:
				# CSVのヘッダーが'clean_path' ではなく'clean' の可能性があるため調整 (元のDataset/generate_reverb_dataset.pyでは'clean_path'だが、ここではフォールバックの柔軟性を考慮)
				if col == "clean_path" and "clean" in self.data_df.columns:
					self.teacher_column = "clean"
					continue

				# 残響特徴量の列がなかった場合、エラーで停止
				print(f"エラー: CSVに必須な列'{col}' が見つかりません。")
				sys.exit(1)

		# --- 欠損値（空のパス）を持つ行を削除 ---
		original_len = len(self.data_df)
		subset_cols = [self.teacher_column, self.input_column] + self.reverb_feature_columns
		self.data_df.dropna(subset=subset_cols, inplace=True)
		self.data_df = self.data_df[(self.data_df[self.teacher_column] != "") & (self.data_df[self.input_column] != "")]

		if len(self.data_df) < original_len:
			print(f"注意: {original_len - len(self.data_df)}行のデータパスに欠損があったため、除外されました。")

		print(f"{csv_path} から {len(self.data_df)} 件のファイルペアを読み込みました。")
		print(f"  - 入力データ: '{self.input_column}' 列, 教師音声: '{self.teacher_column}' 列を使用")
		print(f"  - 教師残響特徴量 {self.reverb_feature_columns} を連結して使用")

	def __getitem__(self, index):
		"""
		指定されたインデックスのデータをロードし、波形ベクトルと補助的な残響特徴量テンソルを構築して返す。

		残響特徴量リストの指定に応じて、文字列リストからPython配列への変換、
		またはスカラーのテンソル化を行い、それら全てを連結して1つの特徴量テンソルとします。

		Args:
			index (int): 取得対象のデータインデックス

		Returns:
			tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
				- noisy_waveform (torch.Tensor): 入力音声
				- clean_waveform (torch.Tensor): 教師音声
				- reverb_feature_tensor (torch.Tensor): パースされた補助残響特徴量の連結テンソル
		"""
		# --- 1. ファイルパスの取得---
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

		# --- 3. 教師残響特徴量のロードと連結---
		feature_tensors = []
		for col in self.reverb_feature_columns:
			feature_value = row[col]

			try:
				if col == "cepstrum_coeffs":
					# 文字列化されたリストを ast.literal_eval でPythonリストに変換
					feature_list = ast.literal_eval(feature_value)
					feature_np = np.array(feature_list, dtype=np.float32)
					feature_tensor = torch.from_numpy(feature_np)  # 例: [16]次元Eクトル
				else:
					# RT60, C50, D50などのスカラー値をテンソル化
					feature_tensor = torch.tensor([float(feature_value)], dtype=torch.float32)  # 例: [1]次元カラー
			except Exception as e:
				# データの破損や形式エラーに対応
				# cepstrum_coeffsEEPC=16EE16次元他EスカラーEE次元と仮宁E
				dim = 16 if col == "cepstrum_coeffs" else 1
				feature_tensor = torch.zeros(dim, dtype=torch.float32)

			feature_tensors.append(feature_tensor)

		# 全ての教師特徴量を連結(例: [16] + [1] + [1] + [1] -> [19])
		reverb_feature_tensor = torch.cat(feature_tensors, dim=0)

		# 変更: 教師残響特徴量(reverb_feature_tensor) をリターンに追加
		return noisy_waveform, clean_waveform, reverb_feature_tensor

	def __len__(self):
		return len(self.data_df)

	@staticmethod
	def collate_fn(batch):
		"""
		波形テンソルの時間長サイズを揃え、さらに残響の補助特徴量テンソルもスタックするcollate関数。

		Args:
			batch (list): __getitem__ で取得された要素（入力波形, 教師波形, 残響特徴量）のタプルのリスト

		Returns:
			tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
				- mix_data (torch.Tensor): 長さを揃えてパディングされた入力波形テンソル [B, 1, T]
				- target_data (torch.Tensor): 長さを揃えてパディングされた教師波形テンソル [B, 1, T]
				- reverb_features_batch (torch.Tensor): スタックされた残響特徴量テンソル [B, N]
		"""
		# バッチ内の最大長を見つける
		max_len = max([x[0].size(-1) for x in batch])

		# 全てのテンソルを最大長にパディング
		padded_batch = []
		reverb_features = []

		# 変更: バッチかめEつの要素 (波形2つと特徴量つ) をアンパック
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

		# 変更: 3つのバッチ化されたテンソルを返す
		return mix_data, target_data, reverb_features_batch




# ===================================================================
# ▼▼▼ 使い方サンプルコード！▼▼▼
# ===================================================================
if __name__ == "__main__":
	# --- このスクリプトを直接実行した際のテストコード---

	# 1. テスト用のCSVファイルを作成 (実際には既存ECSVを使用
	print("--- テスト用のCSVファイルを作成しています---")
	dummy_csv_path = "test_data.csv"
	dummy_data = {
		"clean": ["clean_a.wav", "clean_b.wav", "clean_c.wav"],
		"noise_only": ["noise_a.wav", "noise_b.wav", ""],  # cは欠損
		"noise_reverb": ["noise_reverb_a.wav", "noise_reverb_b.wav", "noise_reverb_c.wav"],
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
	print("--- テストファイルの準備完了---\n")

	# 2. データセットのインスタンスを作成
	print("--- データセットのインスタンスを作成しまぁE---")
	# 入力として「雑音＋残響」E列を持つ
	input_header = "noise_reverb"
	train_dataset = CsvDataset(csv_path=dummy_csv_path, input_column_header=input_header)

	# 3. DataLoaderを作成
	from torch.utils.data import DataLoader

	train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

	# 4. データからバッチを取り出して形状を確認
	print("\n--- DataLoaderからデータを取り出します---")
	# `iter`でイチEータを作成し、`next`で最初のバッチを取得
	noisy_signal, clean_signal = next(iter(train_loader))

	print(f"取得したデータの形状:")
	print(f"  - 入力信号 (Noisy signal): {noisy_signal.shape}")
	print(f"  - 教師信号 (Clean signal): {clean_signal.shape}")

	# 形状の解説: (バッチサイズ, 周波数ビン数, 時間フレーム数)
	# 周波数ビン数 = n_fft / 2 + 1 = 512 / 2 + 1 = 257
	# 時間フレーム数 = chunk_size / hop_length = (16384 * 4) / 128 = 512

	# --- 入力を変更してテスト---
	print("\n--- 入力を変更して再度テストします---")
	input_header_2 = "noise_only"
	train_dataset_2 = CsvDataset(csv_path=dummy_csv_path, input_column_header=input_header_2)
	# noise_c.wavが欠損しているため、データ数は2件になるはず
	assert len(train_dataset_2) == 2, "欠損データが正しく除外されていません"
	print("欠損データの除外を正しく確認しました。")
