import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path

from numba.cuda import const
from tqdm import tqdm
from tqdm.contrib import tenumerate
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import Optional, Tuple, Callable

# 必要なモジュールのインポート
# 🚨 実行環境に応じてパスを調整してください 🚨
try:
	# models/graph_utils.py からインポート
	from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType
	# models/check_Speq_GNN.py からインポート
	from models.check_SpeqGNN import CheckSpeqGNN
	# CsvDataset.py のロジックを流用 (今回はスクリプト内に定義)

	# mymodule/confirmation_GPU.py からインポート
	from mymodule import confirmation_GPU
	# mymodule/my_func.py からインポート (主にディレクトリ作成用)
	from mymodule import my_func, const
except ImportError as e:
	print(f"エラー: 必要なモジュールのインポートに失敗しました。パスを確認してください: {e}", file=sys.stderr)
	sys.exit(1)


# --- 1. 検証用データローダーの定義 ---

# CsvSpectralDataset.py のロジックをベースに、必要な全てのスペクトログラムを返すように拡張
class CheckSpectralDataset(Dataset):
	"""
	CheckSpeqGNNモデルの推論に必要な、ノイズあり/クリーン両方のスペクトログラムを返すデータローダー。
	"""

	def __init__(self, csv_path, input_column_header, sample_rate=16000, max_length_sec=None, n_fft=512, hop_length=256,
	             win_length=None, device='cpu'):
		self.device = device
		self.teacher_column = "clean"
		self.input_column = input_column_header
		self.sample_rate = sample_rate
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.win_length = win_length if win_length is not None else n_fft

		self.max_length_samples = max_length_sec * sample_rate if max_length_sec is not None else None

		# 複素スペクトログラム変換器の初期化
		self.stft_transform_complex = torchaudio.transforms.Spectrogram(
			n_fft=n_fft, hop_length=hop_length, win_length=self.win_length,
			window_fn=torch.hann_window, power=None, return_complex=True,
		)

		# CSVファイルの読み込みと検証
		try:
			self.data_df = pd.read_csv(csv_path)
		except FileNotFoundError:
			raise FileNotFoundError(f"❌ エラー: CSVファイルが見つかりません: {csv_path}")

		if self.teacher_column not in self.data_df.columns or self.input_column not in self.data_df.columns:
			raise ValueError(f"❌ エラー: CSVに必要な列 ('{self.teacher_column}' または '{self.input_column}') が見つかりません。")

		self.data_df.dropna(subset=[self.teacher_column, self.input_column], inplace=True)
		self.data_df = self.data_df[(self.data_df[self.teacher_column] != "") & (self.data_df[self.input_column] != "")]
		print(f"✅ [検証用] {csv_path} から {len(self.data_df)} 件のファイルペアを読み込みました。")

	def __getitem__(self, index):
		row = self.data_df.iloc[index]
		clean_path = Path(row[self.teacher_column])
		noisy_path = Path(row[self.input_column])

		# 音声波形の読み込み (モノラルを想定)
		clean_waveform, _ = torchaudio.load(clean_path)
		noisy_waveform, _ = torchaudio.load(noisy_path)

		# 複数チャンネルの場合は最初のチャンネルを選択
		if noisy_waveform.shape[0] > 1: noisy_waveform = noisy_waveform[0].unsqueeze(0)
		if clean_waveform.shape[0] > 1: clean_waveform = clean_waveform[0].unsqueeze(0)

		# 波形長の調整
		min_len = min(noisy_waveform.shape[-1], clean_waveform.shape[-1])
		if self.max_length_samples is not None:
			min_len = min(min_len, self.max_length_samples)

		noisy_waveform = noisy_waveform[:, :min_len]
		clean_waveform = clean_waveform[:, :min_len]
		noisy_length = noisy_waveform.shape[-1]

		# STFTのためにチャンネル次元を削除 [1, L] -> [L]
		noisy_waveform_squeezed = noisy_waveform.squeeze(0)
		clean_waveform_squeezed = clean_waveform.squeeze(0)

		# STFT適用 (CPU上で実行)
		noisy_complex_spec = self.stft_transform_complex(noisy_waveform_squeezed)
		clean_complex_spec = self.stft_transform_complex(clean_waveform_squeezed)

		# 振幅スペクトログラムを計算し、チャネル次元(1)のみを追加 (F, T -> 1, F, T)
		# 不要な .unsqueeze(0) を削除し、チャネル次元の挿入を一度だけにします。

		# 修正後のコード:
		noisy_magnitude_spec = torch.abs(noisy_complex_spec).unsqueeze(0)  # [1, F, T]
		clean_magnitude_spec = torch.abs(clean_complex_spec).unsqueeze(0)  # [1, F, T]

		noisy_length = noisy_waveform.shape[-1]
		clean_length = clean_waveform.shape[-1]

		file_name = clean_path.stem

		# ノイズあり振幅[1, F, T]、クリーン振幅[1, F, T]、ノイズあり複素[F, T]、元の長さ[int]、ファイル名[str]を返す
		return noisy_magnitude_spec, clean_magnitude_spec, noisy_complex_spec, clean_complex_spec, noisy_length, clean_length, file_name

	def __len__(self):
		return len(self.data_df)


# --- 2. 実行スクリプトの本体 ---

def run_analysis(
		model_path: str,
		test_csv_path: str,
		output_dir: str,
		gnn_type: str,
		num_node: int,
		max_length_sec: Optional[int],
		stft_params: dict,
		csv_input_column: str,
):
	device = confirmation_GPU.get_device()
	print(f"分析に使用するデバイス: {device}")

	# モデル設定
	graph_config = GraphConfig(
		num_edges=num_node,
		node_selection=NodeSelectionType.ALL,
		edge_selection=EdgeSelectionType.KNN,
		bidirectional=True,
	)

	# モデルのインスタンス化
	model = CheckSpeqGNN(
		n_channels=1,
		n_classes=1,
		gnn_type=gnn_type,
		graph_config=graph_config,
		**stft_params,
	).to(device)

	# 学習済み重みのロード
	try:
		model_name = Path(model_path).stem
		# 一般的な「最良モデル」の命名規則を優先してロードを試みる
		best_model_path = Path(model_path).parent / f"BEST_{model_name}.pth"
		loaded_state_dict = torch.load(best_model_path, map_location=device)

		# 冗長なキー名（例: 'module.' プレフィックス）の削除
		if list(loaded_state_dict.keys())[0].startswith('module.'):
			loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}

		model.load_state_dict(loaded_state_dict)
		print(f"✅ 学習済みモデルの重みを {best_model_path.name} からロードしました。")
	except Exception as e:
		print(f"⚠️ 警告: モデルの重みロード中にエラーが発生しました（{e}）。ランダムな重みで続行します。")

	# データローダーの準備
	dataloader = DataLoader(
		CheckSpectralDataset(
			csv_path=test_csv_path,
			input_column_header=csv_input_column,
			max_length_sec=max_length_sec,
			**stft_params,
			device=device,
		),
		batch_size=1,
		shuffle=False
	)

	# --- データ収集の実行 ---
	model.eval()
	all_node_losses = []
	node_connection_counts = defaultdict(int)
	num_nodes_per_file = None
	total_files = 0

	# U-Netのダウンサンプリング係数
	downsample_factor = 2 ** 3
	# ボトルネック層の周波数ビン数
	estimated_freq_bins_bottleneck = int(np.ceil((stft_params['n_fft'] // 2 + 1) / downsample_factor))

	with torch.no_grad():
		print("ノード誤差とエッジ接続回数の収集を開始...")
		# tqdm(dataloader, ...) の代わりにenumerateを使用し、データ収集中にエラーハンドリングを強化
		for i, batch in tenumerate(dataloader):
			# print(batch)
			# print(len(batch))
			# プログレスバーの更新
			if i % 50 == 0 or i == len(dataloader) - 1:
				tqdm.write(f"Collecting Node Metrics: {i}/{len(dataloader)}")

			# --- 1. データローダーからの要素を明示的にアンパック（5要素を前提） ---
			# 5つの要素を明示的にアンパック
			noisy_mag, clean_mag, noisy_complex, clean_complex, noisy_length, clean_length, file_name = batch

			# TensorをPythonのintに変換
			noisy_length_int = noisy_length.item()
			clean_length_int = clean_length.item()
			# ファイル名リストから文字列を取得
			file_name_str = file_name[0] if isinstance(file_name, list) else file_name

			# CUDAに移動
			noisy_mag = noisy_mag.to(device)
			clean_mag = clean_mag.to(device)
			noisy_complex = noisy_complex.to(device)
			clean_complex = clean_complex.to(device)

			# --- 2. モデルのフォワードパスの実行 ---
			# モデル呼び出し。引数は4つで、original_lengthはintに変換済み
			_, noisy_node, noisy_index = model(noisy_mag, noisy_complex, noisy_length_int)
			_, clean_node, clean_index = model(clean_mag, clean_complex, clean_length_int)

			# --- 3. Excelに出力
			output_path = f"{output_dir}/{file_name_str}.xlsx"
			my_func.make_dir(os.path.dirname(output_path))
			# Excelファイルへの書き出し
			with pd.ExcelWriter(output_path) as writer:
				pd.DataFrame(noisy_node.cpu().numpy()).to_excel(writer, sheet_name="node", startcol=0)
				pd.DataFrame(clean_node.cpu().numpy()).to_excel(writer, sheet_name="node", startcol=noisy_node.shape[1] + 1)
				pd.DataFrame(noisy_index.cpu().numpy().T).to_excel(writer, sheet_name="noisy_index")
				pd.DataFrame(clean_index.cpu().numpy().T).to_excel(writer, sheet_name="clean_index")



	print(f"\n=======================================================")
	print(f"✅ GNNノード分析が完了しました。")
	print(f"=======================================================")


# --- 3. 実行ブロック ---

if __name__ == "__main__":
	# 🚨🚨🚨 以下を**必ず**あなたの環境に合わせて修正してください 🚨🚨🚨

	# 1. モデルファイルの設定
	# モデルの重みファイルが保存されているディレクトリとファイル名
	model = "SpeqGAT"
	wave_type = "noise_reverb"
	speech_type = "DEMAND_DEMAND"
	MODEL_BASE_DIR = f"{const.PTH_DIR}/{speech_type}/{model}"  # 例: "models/saved_models/SpeqGAT_noise_only"
	MODEL_NAME = f"{model}_{wave_type}"
	MODEL_PATH = f"{MODEL_BASE_DIR}/SISDR_SpeqGAT_{wave_type}_32node_all_knn.pth"  # 例: BEST_SpeqGAT_noise_only.pthをロード
	# 2. データセットCSVファイルのパス
	CSV_PATH = Path(f"{const.MIX_DATA_DIR}/{speech_type}/test.csv")
	# 3. 出力ディレクトリ
	OUTPUT_DIR = f"{const.OUTPUT_WAV_DIR}/{speech_type}/{model}/gnn_node_analysis"

	# 4. モデルのハイパーパラメータ設定 (学習時と一致させる必要があります)
	GNN_TYPE = "GAT"
	NUM_NODE_EDGES = 32
	MAX_LENGTH_SEC = None
	CSV_INPUT_COL = "noise_reverb"

	STFT_PARAMS = {
		"n_fft": 512,
		"hop_length": 256,
		"win_length": 512,
	}

	print("--- GNNノード分析プログラムの実行 ---")

	# 実行
	run_analysis(
		model_path=Path(MODEL_PATH),
		test_csv_path=str(CSV_PATH),
		output_dir=OUTPUT_DIR,
		gnn_type=GNN_TYPE,
		num_node=NUM_NODE_EDGES,
		max_length_sec=MAX_LENGTH_SEC,
		stft_params=STFT_PARAMS,
		csv_input_column=CSV_INPUT_COL,
	)
