import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import glob  # ファイルパスのパターンマッチングに便利

from mymodule import my_func

class AudioDataset(Dataset):
    def __init__(self, noisy_audio_dir, clean_audio_dir, sample_rate=16000, max_length_sec=5):
        """
        オーディオデータセットクラス

        Args:
            noisy_audio_dir (str): 雑音を含む音声ファイルが格納されたディレクトリのパス。
            clean_audio_dir (str): クリーンな音声ファイルが格納されたディレクトリのパス。
            sample_rate (int): 音声データをリサンプリングするターゲットのサンプリングレート。
            max_length_sec (int): 処理する音声の最大長（秒）。これより長い音声は切り捨てられる。
                                  Noneの場合、最大長の制限なし。
        """
        self.noisy_audio_dir = noisy_audio_dir
        self.clean_audio_dir = clean_audio_dir
        self.sample_rate = sample_rate
        self.max_length_samples = max_length_sec * sample_rate if max_length_sec is not None else None

        # 雑音を含む音声ファイルのリストを取得
        # 例えば、.wav ファイルのみを対象とする
        self.noisy_file_paths = sorted(glob.glob(os.path.join(noisy_audio_dir, "*.wav")))

        # クリーンな音声ファイルのリストを取得
        self.clean_file_paths = sorted(glob.glob(os.path.join(clean_audio_dir, "*.wav")))

        # ファイル数の一致を確認（重要なチェック）
        if len(self.noisy_file_paths) != len(self.clean_file_paths):
            raise ValueError("The number of noisy and clean audio files does not match.")

        # ファイル名のペアリングを確認（これも重要）
        for i in range(len(self.noisy_file_paths)):
            # noisy_filename = os.path.basename(self.noisy_file_paths[i])
            # clean_filename = os.path.basename(self.clean_file_paths[i])
            noisy_filename = my_func.get_file_name(self.noisy_file_paths[i])[0]
            clean_filename = my_func.get_file_name(self.clean_file_paths[i])[0]
            if not(noisy_filename in clean_filename):
                raise ValueError(f"Mismatched filenames: {noisy_filename} != {clean_filename} at index {i}")

        print(f"Found {len(self.noisy_file_paths)} audio pairs.")

    def __len__(self):
        return len(self.noisy_file_paths)

    def __getitem__(self, idx):
        # 雑音を含む音声の読み込み
        noisy_path = self.noisy_file_paths[idx]
        noisy_waveform, current_sample_rate = torchaudio.load(noisy_path)

        # クリーンな音声の読み込み
        clean_path = self.clean_file_paths[idx]
        clean_waveform, _ = torchaudio.load(clean_path)  # サンプリングレートはnoisy_waveformと同じはず

        # サンプリングレートのリサンプリング
        if current_sample_rate != self.sample_rate:
            noisy_waveform = torchaudio.transforms.Resample(current_sample_rate, self.sample_rate)(noisy_waveform)
            clean_waveform = torchaudio.transforms.Resample(current_sample_rate, self.sample_rate)(clean_waveform)

        # チャンネル数の調整（例：ステレオ -> モノラル）
        # モデルが1チャンネル入力を想定している場合、モノラルに変換
        if noisy_waveform.shape[0] > 1:
            noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)
        if clean_waveform.shape[0] > 1:
            clean_waveform = torch.mean(clean_waveform, dim=0, keepdim=True)

        # 長さの調整
        # (1) 最大長に切り捨て
        if self.max_length_samples is not None and noisy_waveform.shape[1] > self.max_length_samples:
            noisy_waveform = noisy_waveform[:, :self.max_length_samples]
            clean_waveform = clean_waveform[:, :self.max_length_samples]

        # (2) パディング（短いサンプルを埋める）
        # このモデルは固定長入力を必要としないが、バッチ処理のために長さを揃える必要がある場合
        # または、常に同じ長さのサンプルを入力したい場合は、ここでパディングを行う
        # 例:
        # if self.max_length_samples is not None and noisy_waveform.shape[1] < self.max_length_samples:
        #     padding_amount = self.max_length_samples - noisy_waveform.shape[1]
        #     noisy_waveform = F.pad(noisy_waveform, (0, padding_amount))
        #     clean_waveform = F.pad(clean_waveform, (0, padding_amount))

        # 正規化（オプション）：-1から1の範囲に正規化されていることがほとんどだが、確認
        # ピーク値で正規化
        noisy_waveform = noisy_waveform / (noisy_waveform.abs().max() + 1e-8)
        clean_waveform = clean_waveform / (clean_waveform.abs().max() + 1e-8)

        # モデルの入力は [batch, n_channels, length]
        # ターゲットも [batch, n_channels, length]
        # print("dataset_out:", noisy_waveform.shape)
        # print("dataset_out:", clean_waveform.shape)
        return noisy_waveform, clean_waveform


# --- 使用例 ---
if __name__ == "__main__":
    # テスト用のダミーデータを作成 (実際にはwavファイルを配置)
    test_noisy_dir = "test_noisy_audio"
    test_clean_dir = "test_clean_audio"
    os.makedirs(test_noisy_dir, exist_ok=True)
    os.makedirs(test_clean_dir, exist_ok=True)

    test_sample_rate = 16000
    test_length_samples = test_sample_rate * 3  # 3秒の音声

    print("Creating dummy audio files...")
    for i in range(5):  # 5つのダミー音声ファイルペアを作成
        if i < 3:
            test_length_samples = test_sample_rate * 3  # 3秒の音声
        else:
            test_length_samples = test_sample_rate * 5  # 3秒の音声

        # ダミーのノイズ音声（ランダムノイズ）
        dummy_noisy_waveform = torch.randn(1, test_length_samples) * 0.5
        torchaudio.save(os.path.join(test_noisy_dir, f"audio_{i:03d}.wav"), dummy_noisy_waveform, test_sample_rate)

        # ダミーのクリーン音声（サイン波）
        t = torch.linspace(0, test_length_samples / test_sample_rate, test_length_samples)
        dummy_clean_waveform = 0.8 * torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440Hzサイン波
        torchaudio.save(os.path.join(test_clean_dir, f"audio_{i:03d}.wav"), dummy_clean_waveform, test_sample_rate)
    print("Dummy audio files created.")

    # データセットの初期化
    dataset = AudioDataset(
        noisy_audio_dir=test_noisy_dir,
        clean_audio_dir=test_clean_dir,
        sample_rate=test_sample_rate,
        max_length_sec=5  # 最大5秒にクリップ
    )

    # データローダーの初期化
    # バッチ処理のために長さを揃える必要がある場合は、collate_fnを実装するか、
    # __getitem__内でパディングする必要があります。
    # 現在のモデルは固定長ではないですが、異なる長さのサンプルをバッチにまとめる場合は注意が必要です。
    # 最も単純なアプローチは、DataLoaderのbatch_size=1で常に1サンプルずつ処理するか、
    # もしくは全サンプルを同じmax_length_samplesにパディングすることです。

    # ここでは、簡単のためにbatch_size=1で試す
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)  # num_workersはデバッグ中は0が推奨

    print("\nIterating through DataLoader:")
    for i, (noisy_audio, clean_audio) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"  Noisy audio shape: {noisy_audio.shape}")  # [B, C, L]
        print(f"  Clean audio shape: {clean_audio.shape}")  # [B, C, L]

        # ここでモデルのフォワードパスを実行
        # 例: model(noisy_audio.to(device))

        if i >= 2:  # 最初の3バッチだけ表示
            break

    print("\nData loading test complete.")

    # ダミーデータを削除
    import shutil

    shutil.rmtree(test_noisy_dir)
    shutil.rmtree(test_clean_dir)
    print("Dummy audio files removed.")