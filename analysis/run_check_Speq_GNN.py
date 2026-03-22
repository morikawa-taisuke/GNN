# coding:utf-8
import sys
import torch
import h5py # ★ 追加
from typing import Optional
from pathlib import Path
import torchaudio
import pandas as pd

from numba.cuda import const
from tqdm import tqdm
from tqdm.contrib import tenumerate
from torch.utils.data import Dataset, DataLoader

# models/graph_utils.py からインポート
from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType
# models/check_Speq_GNN.py からインポート
from models.check_SpeqGNN import CheckSpeqGNN
# CsvDataset.py のロジックを流用 (今回はスクリプト内に定義)

# mymodule/confirmation_GPU.py からインポート
from mymodule import confirmation_GPU
# mymodule/my_func.py からインポート (主にディレクトリ作成用)
from mymodule import my_func, const


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

		file_name = noisy_path.stem

		# ノイズあり振幅[1, F, T]、クリーン振幅[1, F, T]、ノイズあり複素[F, T]、元の長さ[int]、ファイル名[str]を返す
		return noisy_magnitude_spec, clean_magnitude_spec, noisy_complex_spec, clean_complex_spec, noisy_length, clean_length, file_name

	def __len__(self):
		return len(self.data_df)


# --- 2. 実行スクリプトの本体 ---

def run_analysis(
		model_path: str,
		test_csv_path: str,
		output_dir: str, # ★ HDF5ファイルを含むディレクトリ
		hdf5_filename: str, # ★ 出力HDF5ファイル名
		gnn_type: str,
		num_node: int,
		max_length_sec: Optional[int],
		stft_params: dict,
		csv_input_column: str,
):
	device = confirmation_GPU.get_device()
	print(f"分析に使用するデバイス: {device}")

	# ... (モデル設定、モデルインスタンス化、重みロードは変更なし) ...
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
		best_model_path = Path(model_path).parent / f"{model_name}.pth"
		loaded_state_dict = torch.load(best_model_path, map_location=device)

		# 冗長なキー名（例: 'module.' プレフィックス）の削除
		if list(loaded_state_dict.keys())[0].startswith('module.'):
			loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}

		model.load_state_dict(loaded_state_dict)
		print(f"✅ 学習済みモデルの重みを {best_model_path.name} からロードしました。")
	except Exception as e:
		print(f"⚠️ 警告: モデルの重みロード中にエラーが発生しました（{e}）。ランダムな重みで続行します。")

	# データローダーの準備 (変更なし)
	dataloader = DataLoader(
		CheckSpectralDataset(
			csv_path=test_csv_path,
			input_column_header=csv_input_column,
			max_length_sec=max_length_sec,
			**stft_params,
			device=device, # CheckSpectralDataset に device 引数がない場合は削除
		),
		batch_size=1, # ★ バッチサイズ1を維持 (HDF5への書き込みロジックがバッチ=1前提のため)
		shuffle=False,
		num_workers=4 # ★ データ読み込み高速化のため num_workers を追加 (環境に合わせて調整)
	)

	# --- HDF5ファイルを開く ---
	output_hdf5_path = Path(output_dir) / hdf5_filename
	my_func.make_dir(str(output_hdf5_path)) # 出力ディレクトリを作成
	hdf5_file = h5py.File(output_hdf5_path, 'w')
	print(f"✅ 結果を {output_hdf5_path} に出力します。")


	# --- データ収集の実行 ---
	model.eval()
	# all_node_losses や node_connection_counts はこのスクリプトでは使われていないようなのでコメントアウト
	# all_node_losses = []
	# node_connection_counts = defaultdict(int)
	# num_nodes_per_file = None
	# total_files = 0

	# U-Netのダウンサンプリング係数 (もし使うなら残す)
	# downsample_factor = 2 ** 3
	# estimated_freq_bins_bottleneck = int(np.ceil((stft_params['n_fft'] // 2 + 1) / downsample_factor))

	try: # ★ ファイルI/Oエラー等に備えて try...finally を追加
		with torch.no_grad():
			print("ノード特徴量とエッジ情報の収集を開始...")
			for i, batch in tenumerate(dataloader):
				# if i % 50 == 0 or i == len(dataloader) - 1:
				# 	tqdm.write(f"Processing: {i}/{len(dataloader)}")

				# --- 1. データローダーからの要素をアンパック ---
				noisy_mag, clean_mag, noisy_complex, clean_complex, noisy_length, clean_length, file_name = batch

				noisy_length_int = noisy_length.item()
				clean_length_int = clean_length.item()
				# ファイル名はリストの場合があるので最初の要素を取得
				file_name_str = file_name[0] if isinstance(file_name, (list, tuple)) else file_name
				# print(file_name_str) # デバッグ用

				# exit(2) # 元のコードにあった exit を削除

				# CUDAに移動
				noisy_mag = noisy_mag.to(device)
				clean_mag = clean_mag.to(device)
				noisy_complex = noisy_complex.to(device)
				clean_complex = clean_complex.to(device)

				# --- 2. モデルのフォワードパスの実行 ---
				# ★変更点: 戻り値が増えたので bottleneck_shape を受け取る
				_, noisy_node, noisy_index, bottleneck_shape = model(noisy_mag, noisy_complex, noisy_length_int)
				# clean側は形状が同じなので shape は受け取らなくて良い ( _ で捨てる)
				_, clean_node, clean_index, _ = model(clean_mag, clean_complex, clean_length_int)

				# --- 3. HDF5ファイルに出力 ---
				# ファイル名をキーにしたグループを作成 (存在すれば上書き)
				file_group = hdf5_file.create_group(file_name_str)

				# ★追加点: 形状情報を属性(metadata)として保存
				file_group.attrs['bottleneck_height'] = bottleneck_shape[0]  # 周波数ビン数
				file_group.attrs['bottleneck_width'] = bottleneck_shape[1]  # 時間フレーム数

				# 各データをNumPy配列に変換してデータセットとして保存
				file_group.create_dataset("noisy_node", data=noisy_node.cpu().numpy())
				file_group.create_dataset("clean_node", data=clean_node.cpu().numpy())
				file_group.create_dataset("error_node", data=(clean_node - noisy_node).cpu().numpy())
				file_group.create_dataset("noisy_index", data=noisy_index.cpu().numpy().T) # 元のExcel出力に合わせて転置
				file_group.create_dataset("clean_index", data=clean_index.cpu().numpy().T) # 元のExcel出力に合わせて転置

	finally: # ★ ループ終了後またはエラー発生時にファイルを確実に閉じる
		if 'hdf5_file' in locals() and hdf5_file:
			hdf5_file.close()
			print(f"✅ HDF5ファイル {output_hdf5_path} を閉じました。")


	print(f"\n=======================================================")
	print(f"✅ GNNノード分析が完了しました。")
	print(f"=======================================================")


# --- 3. 実行ブロック ---

if __name__ == "__main__":
	# 🚨🚨🚨 以下を**必ず**あなたの環境に合わせて修正してください 🚨🚨🚨

	# 1. モデルファイルの設定
	model = "SpeqGAT"
	wave_type = "noise_reverb"
	speech_type = "DEMAND_DEMAND"
	MODEL_BASE_DIR = f"{const.CHECKPOINT_DIR}/{speech_type}/{model}"
	MODEL_NAME = f"{model}_{wave_type}"
	# ★ 必要に応じてモデルパスを修正
	MODEL_PATH = f"{MODEL_BASE_DIR}/BEST_new_{model}_{wave_type}e_16node_temporal_knn.pth" # BESTモデルを使うことを推奨

	# 2. データセットCSVファイルのパス
	CSV_PATH = Path(f"{const.MIX_DATA_DIR}/{speech_type}/test.csv")

	# 3. 出力ディレクトリとHDF5ファイル名
	OUTPUT_DIR = f"{const.OUTPUT_WAV_DIR}/{speech_type}/{model}/gnn_node_analysis_hdf5/{wave_type}" # ★ 出力ディレクトリ変更
	HDF5_FILENAME = f"{MODEL_NAME}_16node_temporal_knn.h5" # ★ HDF5ファイル名を設定

	# 4. モデルのハイパーパラメータ設定 (学習時と一致させる)
	GNN_TYPE = "GAT"
	NUM_NODE_EDGES = 16
	MAX_LENGTH_SEC = None
	CSV_INPUT_COL = "noise_reverb" # ★ CSV内の入力列名

	STFT_PARAMS = {
		"n_fft": 512,
		"hop_length": 256,
		"win_length": 512,
	}

	print("--- GNNノード分析プログラムの実行 (HDF5出力) ---")

	# 実行 (引数を更新)
	run_analysis(
		model_path=Path(MODEL_PATH),
		test_csv_path=str(CSV_PATH),
		output_dir=OUTPUT_DIR,       # ★
		hdf5_filename=HDF5_FILENAME, # ★
		gnn_type=GNN_TYPE,
		num_node=NUM_NODE_EDGES,
		max_length_sec=MAX_LENGTH_SEC,
		stft_params=STFT_PARAMS,
		csv_input_column=CSV_INPUT_COL,
	)