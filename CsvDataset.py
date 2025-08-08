import sys

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


# ===================================================================
# ▼▼▼ 改良版データローダ ▼▼▼
# ===================================================================


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

    def __init__(self, csv_path, input_column_header, chunk_size=16384 * 4, sample_rate=16000, max_length_sec=3):

        super(CsvDataset, self).__init__()

        self.chunk_size = chunk_size
        self.teacher_column = "clean"  # 教師データは常に 'clean_path' を使用
        self.input_column = input_column_header

        self.max_length_samples = max_length_sec * sample_rate

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
        clean_path = row[self.teacher_column]
        noisy_path = row[self.input_column]

        clean_waveform, current_sample_rate = torchaudio.load(clean_path)
        noisy_waveform, _ = torchaudio.load(noisy_path)

        if noisy_waveform.shape[-1] > self.max_length_samples:
            noisy_waveform = noisy_waveform[:, : self.max_length_samples]
            clean_waveform = clean_waveform[:, : self.max_length_samples]
        elif noisy_waveform.shape[-1] < self.max_length_samples:
            padding_amount = self.max_length_samples - noisy_waveform.shape[1]
            noisy_waveform = F.pad(noisy_waveform, (0, padding_amount))
            clean_waveform = F.pad(clean_waveform, (0, padding_amount))
        return noisy_waveform, clean_waveform

    def __len__(self):
        """
        データセットの総数を返す。
        """
        return len(self.data_df)


# ===================================================================
# ▼▼▼ 使い方（サンプルコード） ▼▼▼
# ===================================================================
if __name__ == "__main__":
    # --- このスクリプトを直接実行した際のテストコード ---

    # 1. テスト用のCSVファイルを作成 (実際には既存のCSVを使う)
    print("--- テスト用のCSVファイルを作成しています ---")
    dummy_csv_path = "test_data.csv"
    dummy_data = {
        "clean_path": ["clean_a.wav", "clean_b.wav", "clean_c.wav"],
        "noise_only_path": ["noise_a.wav", "noise_b.wav", ""],  # cは欠損
        "noise_reverb_path": ["noise_reverb_a.wav", "noise_reverb_b.wav", "noise_reverb_c.wav"],
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
    input_header = "noise_reverb_path"
    train_dataset = CsvDataset(csv_path=dummy_csv_path, input_column_header=input_header)

    # 3. DataLoaderを作成
    from torch.utils.data import DataLoader

    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

    # 4. データを1バッチ取り出して形状を確認
    print("\n--- DataLoaderからデータを取り出します ---")
    # `iter`でイテレータを作成し、`next`で最初のバッチを取得
    noisy_mag, noisy_phase, clean_mag = next(iter(train_loader))

    print(f"取得したデータの形状:")
    print(f"  - 入力マグニチュード (Noisy Mag): {noisy_mag.shape}")
    print(f"  - 入力フェーズ (Noisy Phase): {noisy_phase.shape}")
    print(f"  - 教師マグニチュード (Clean Mag): {clean_mag.shape}")

    # 形状の解説: (バッチサイズ, 周波数ビン数, 時間フレーム数)
    # 周波数ビン数 = n_fft / 2 + 1 = 512 / 2 + 1 = 257
    # 時間フレーム数 = chunk_size / hop_length = (16384 * 4) / 128 = 512

    # --- 入力列を変更してテスト ---
    print("\n--- 入力列を変更して再度テストします ---")
    input_header_2 = "noise_only_path"
    train_dataset_2 = CsvDataset(csv_path=dummy_csv_path, input_column_header=input_header_2)
    # noise_c.wavが欠損しているため、データ数は2件になるはず
    assert len(train_dataset_2) == 2, "欠損データが正しく除外されていません"
    print("✅ 欠損データの除外を正しく確認しました。")
