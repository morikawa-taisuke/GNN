import sys
import os
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf

# ===================================================================
# ▼▼▼ 【学習・検証用】周波数領域データローダ ▼▼▼
# ===================================================================
class CsvSpectralDataset(Dataset):
    """
    CSVファイルからファイルパスを読み込み、音声データをスペクトログラムに変換して
    ロードするDatasetクラス（学習・検証用）。

    Args:
        csv_path (str): データセットのパス情報が記載されたCSVファイルのパス。
        input_column_header (str): 入力データとして使用するCSVの列名。
        sample_rate (int): 音声データのサンプリングレート（Hz）。
        max_length_sec (int): 音声データの最大長（秒）。これを超える場合は切り捨て/短い場合はパディング。
        n_fft (int): FFTのポイント数。
        hop_length (int): STFTのホップ長。
        win_length (int): STFTの窓長。Noneの場合、n_fftと同じ値が使われる。
    """

    def __init__(self, csv_path, input_column_header, sample_rate=16000, max_length_sec=3, n_fft=512, hop_length=256, win_length=None):
        super(CsvSpectralDataset, self).__init__()

        self.teacher_column = "clean"  # 教師データは 'clean' 列を想定
        self.input_column = input_column_header
        self.sample_rate = sample_rate
        self.max_length_samples = max_length_sec * sample_rate

        # --- STFT変換の定義 ---
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length if win_length is not None else n_fft,
            window_fn=torch.hann_window,
            power=1.0,  # 振幅スペクトログラム (Magnitude)
        )

        # --- CSVファイルの読み込みと検証 ---
        try:
            self.data_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ エラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
            sys.exit(1)

        if self.teacher_column not in self.data_df.columns or self.input_column not in self.data_df.columns:
            print(f"❌ エラー: CSVに必要な列 ('{self.teacher_column}' または '{self.input_column}') が見つかりません。", file=sys.stderr)
            sys.exit(1)

        original_len = len(self.data_df)
        self.data_df.dropna(subset=[self.teacher_column, self.input_column], inplace=True)
        self.data_df = self.data_df[(self.data_df[self.teacher_column] != "") & (self.data_df[self.input_column] != "")]
        if len(self.data_df) < original_len:
            print(f"⚠️  注意: {original_len - len(self.data_df)}行のデータパスに欠損があったため、除外されました。")

        print(f"✅ [学習用] {csv_path} から {len(self.data_df)} 件のファイルペアを読み込みました。")
        print(f"  - 入力データ: '{self.input_column}' 列, 教師データ: '{self.teacher_column}' 列を使用")

    def __getitem__(self, index):
        """
        指定されたインデックスのデータをロードし、STFTを適用して返す。
        """
        row = self.data_df.iloc[index]
        clean_path = row[self.teacher_column]
        noisy_path = row[self.input_column]

        # --- 音声波形の読み込み ---
        clean_waveform, _ = torchaudio.load(clean_path)
        noisy_waveform, _ = torchaudio.load(noisy_path)

        # --- 波形長の調整（パディング or 切り捨て） ---
        if noisy_waveform.shape[-1] > self.max_length_samples:
            noisy_waveform = noisy_waveform[:, : self.max_length_samples]
            clean_waveform = clean_waveform[:, : self.max_length_samples]
        elif noisy_waveform.shape[-1] < self.max_length_samples:
            padding_amount = self.max_length_samples - noisy_waveform.shape[1]
            noisy_waveform = F.pad(noisy_waveform, (0, padding_amount))
            clean_waveform = F.pad(clean_waveform, (0, padding_amount))

        # --- STFT適用 ---
        noisy_spectrogram = self.stft_transform(noisy_waveform)
        clean_spectrogram = self.stft_transform(clean_waveform)

        return noisy_spectrogram, clean_spectrogram

    def __len__(self):
        return len(self.data_df)


# ===================================================================
# ▼▼▼ 【推論用】周波数領域データローダ ▼▼▼
# ===================================================================
class CsvSpectralInferenceDataset(Dataset):
    """
    推論用に、CSVファイルから入力音声のパスを読み込み、スペクトログラムに変換するDatasetクラス。
    音声再構成のために、複素スペクトログラムも返す。

    Args:
        csv_path (str): データセットのパス情報が記載されたCSVファイルのパス。
        input_column_header (str): 入力データとして使用するCSVの列名。
        sample_rate (int): 音声データのサンプリングレート（Hz）。
        n_fft (int): FFTのポイント数。
        hop_length (int): STFTのホップ長。
        win_length (int): STFTの窓長。Noneの場合、n_fftと同じ値が使われる。
    """
    def __init__(self, csv_path, input_column_header, sample_rate=16000, n_fft=512, hop_length=256, win_length=None):
        super(CsvSpectralInferenceDataset, self).__init__()

        self.input_column = input_column_header
        self.sample_rate = sample_rate

        # --- STFT変換の定義 ---
        # モデル入力用の振幅スペクトログラム
        self.stft_magnitude_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length if win_length is not None else n_fft,
            window_fn=torch.hann_window,
            power=1.0,
        )
        # 音声再構成（ISTFT）用の複素スペactrogram
        self.stft_complex_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length if win_length is not None else n_fft,
            window_fn=torch.hann_window,
            power=None,
            return_complex=True,
        )

        # --- CSVファイルの読み込みと検証 ---
        try:
            self.data_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ エラー: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
            sys.exit(1)

        if self.input_column not in self.data_df.columns:
            print(f"❌ エラー: CSVに入力データ用の列 '{self.input_column}' が見つかりません。", file=sys.stderr)
            sys.exit(1)

        original_len = len(self.data_df)
        self.data_df.dropna(subset=[self.input_column], inplace=True)
        self.data_df = self.data_df[self.data_df[self.input_column] != ""]
        if len(self.data_df) < original_len:
            print(f"⚠️  注意: {original_len - len(self.data_df)}行のデータパスに欠損があったため、除外されました。")

        print(f"✅ [推論用] {csv_path} から {len(self.data_df)} 件の音声ファイルを読み込みました。")
        print(f"  - 入力データ: '{self.input_column}' 列を使用")


    def __getitem__(self, index):
        """
        指定されたインデックスのデータをロードし、STFTを適用して返す。
        """
        row = self.data_df.iloc[index]
        noisy_path = row[self.input_column]

        # --- 音声波形の読み込み ---
        noisy_waveform, current_sample_rate = torchaudio.load(noisy_path)

        # --- リサンプリング（必要に応じて） ---
        if current_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(current_sample_rate, self.sample_rate)
            noisy_waveform = resampler(noisy_waveform)

        original_length = noisy_waveform.shape[-1]

        # --- STFT適用 ---
        noisy_magnitude_spectrogram = self.stft_magnitude_transform(noisy_waveform)
        noisy_complex_spectrogram = self.stft_complex_transform(noisy_waveform)

        # --- ファイル名の取得 ---
        file_name = os.path.splitext(os.path.basename(noisy_path))[0]

        return noisy_magnitude_spectrogram, noisy_complex_spectrogram, original_length, file_name

    def __len__(self):
        return len(self.data_df)


# ===================================================================
# ▼▼▼ 使い方（サンプルコード） ▼▼▼
# ===================================================================
if __name__ == "__main__":
    # --- 1. テスト用のCSVファイルと音声ファイルを作成 ---
    print("\n--- テスト用のダミーファイルを作成しています ---")
    # 学習用CSV
    train_csv_path = "spectral_train_test.csv"
    train_data = {
        "clean": ["clean_a.wav", "clean_b.wav"],
        "noisy": ["noisy_a.wav", "noisy_b.wav"],
    }
    pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
    # 推論用CSV
    inference_csv_path = "spectral_inference_test.csv"
    inference_data = {"input_path": ["noisy_a.wav", "noisy_b.wav"]}
    pd.DataFrame(inference_data).to_csv(inference_csv_path, index=False)

    # ダミー音声ファイル
    sr = 16000
    for name in ["clean_a", "clean_b", "noisy_a", "noisy_b"]:
        sf.write(f"{name}.wav", np.random.randn(sr * 4), sr) # 4秒の音声
    print("--- テストファイルの準備完了 ---\n")


    # --- 2.【学習用】データローダのテスト ---
    print("--- CsvSpectralDataset（学習・検証用）のテスト ---")
    train_dataset = CsvSpectralDataset(
        csv_path=train_csv_path,
        input_column_header="noisy",
        max_length_sec=3 # 3秒に調整
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    noisy_spec, clean_spec = next(iter(train_loader))
    print("取得したデータの形状:")
    print(f"  - 入力スペクトログラム (Noisy): {noisy_spec.shape}")
    print(f"  - 教師スペクトログラム (Clean): {clean_spec.shape}\n")
    # 形状: [Batch, Channel, Freq_bins, Time_frames]


    # --- 3.【推論用】データローダのテスト ---
    print("--- CsvSpectralInferenceDataset（推論用）のテスト ---")
    inference_dataset = CsvSpectralInferenceDataset(
        csv_path=inference_csv_path,
        input_column_header="input_path"
    )
    inference_loader = DataLoader(dataset=inference_dataset, batch_size=2, shuffle=False)
    noisy_mag_spec, noisy_complex_spec, length, f_name = next(iter(inference_loader))
    print("取得したデータの形状と情報:")
    print(f"  - 入力振幅スペクトログラム: {noisy_mag_spec.shape}")
    print(f"  - 入力複素スペクトログラム: {noisy_complex_spec.shape} (dtype: {noisy_complex_spec.dtype})")
    print(f"  - 元の波形長: {length}")
    print(f"  - ファイル名: {f_name}\n")


    # --- 4. テストファイルを削除 ---
    print("--- テストファイルをクリーンアップします ---")
    os.remove(train_csv_path)
    os.remove(inference_csv_path)
    for name in ["clean_a.wav", "clean_b.wav", "noisy_a.wav", "noisy_b.wav"]:
        os.remove(name)
    print("--- クリーンアップ完了 ---")