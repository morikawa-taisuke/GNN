import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import os
import glob
from pathlib import Path
from typing import Optional, Callable, Tuple
from abc import ABC, abstractmethod

# オーディオデータセットの共通処理をカプセル化する基底クラス
class _BaseAudioDataset(Dataset, ABC): # ABCを継承し、抽象メソッドを持つことを明示
    """
    オーディオデータセットの共通処理をカプセル化する基底クラス。
    直接インスタンス化されることはなく、AudioDatasetとAudioDatasetTestによって継承されます。
    """
    def __init__(self, sample_rate: int, max_length: Optional[int], transform: Optional[Callable] = None):
        self.sample_rate = sample_rate
        # 最大長が指定されている場合、サンプル数に変換
        self.max_length_samples = max_length * sample_rate if max_length is not None else None
        self.transform = transform

    def _process_waveform(self, audio_path: Path) -> torch.Tensor:
        """
        単一の音声ファイルをロードし、リサンプリング、長さ調整、変換を行うヘルパーメソッド。

        Args:
            audio_path (Path): 処理する音声ファイルのパス。

        Returns:
            torch.Tensor: 処理された音声波形。
        """
        waveform, current_sample_rate = torchaudio.load(audio_path)

        # 必要に応じてリサンプリング
        if current_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(current_sample_rate, self.sample_rate)(waveform)

        # 音声長の調整 (クリッピングとパディング)
        if self.max_length_samples is not None:
            if waveform.shape[-1] > self.max_length_samples:
                # 最大長より長い場合は切り捨て
                waveform = waveform[:, :self.max_length_samples]
            elif waveform.shape[-1] < self.max_length_samples:
                # 最大長より短い場合はパディング
                padding_amount = self.max_length_samples - waveform.shape[-1]
                waveform = F.pad(waveform, (0, padding_amount))

        # データ変換 (データ拡張など) を適用
        if self.transform:
            waveform = self.transform(waveform)

        return waveform

    @abstractmethod
    def __len__(self) -> int:
        """
        データセットの要素数を返します。サブクラスで実装する必要があります。
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        指定されたインデックスの要素を返します。サブクラスで実装する必要があります。
        """
        pass


class AudioDataset(_BaseAudioDataset):
    """
    オーディオデータセットクラス (学習・検証用)。ノイズありとクリーンな音声ペアをロードします。
    _BaseAudioDatasetを継承し、共通の音声処理ロジックを再利用します。
    """
    def __init__(self, clean_dir: str, noisy_dir: str, sample_rate: int, max_length: Optional[int], transform: Optional[Callable] = None):
        super().__init__(sample_rate, max_length, transform)
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        # ノイズあり・クリーン音声ファイルのパスをソートして取得
        self.noisy_file_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
        self.clean_file_paths = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))

        # ファイル数の不一致またはファイルが見つからない場合のエラーチェック
        if len(self.noisy_file_paths) != len(self.clean_file_paths) or len(self.noisy_file_paths) == 0:
            print(f"noisy_dir: {len(self.noisy_file_paths)} 個")
            print(f"clean_dir: {len(self.clean_file_paths)} 個")
            raise ValueError("ノイズあり音声ファイルとクリーン音声ファイルの数が一致しないか、ファイルが見つかりません。")

    def __len__(self) -> int:
        # データセットのサイズを返す (ノイズありファイルの数に基づく)
        return len(self.noisy_file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        # 指定されたインデックスのファイルパスを取得
        clean_path = Path(self.clean_file_paths[idx])
        noisy_path = Path(self.noisy_file_paths[idx])

        # 基底クラスのヘルパーメソッドを使用して波形を処理
        clean_waveform = self._process_waveform(clean_path)
        noisy_waveform = self._process_waveform(noisy_path)

        # ノイズあり波形、クリーン波形、ノイズありファイルパスを返す
        return noisy_waveform, clean_waveform


class AudioDatasetTest(_BaseAudioDataset):
    """
    オーディオデータセットクラス (テスト用)。ノイズあり音声のみをロードします。
    _BaseAudioDatasetを継承し、共通の音声処理ロジックを再利用します。
    """
    def __init__(self, noisy_dir: str, sample_rate: int = 16000, max_length: Optional[int] = None, transform: Optional[Callable] = None):
        super().__init__(sample_rate, max_length, transform)
        self.noisy_dir = noisy_dir

        # ノイズあり音声ファイルのパスをソートして取得
        self.noisy_file_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))

        # ファイルが見つからない場合のエラーチェック
        if len(self.noisy_file_paths) == 0:
            raise ValueError("指定されたディレクトリに音声ファイルが見つかりません。")

    def __len__(self) -> int:
        # データセットのサイズを返す (ノイズありファイルの数に基づく)
        return len(self.noisy_file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # 指定されたインデックスのファイルパスを取得
        noisy_path = Path(self.noisy_file_paths[idx])
        
        # 基底クラスのヘルパーメソッドを使用して波形を処理
        noisy_waveform = self._process_waveform(noisy_path)

        # ノイズあり波形とファイルパスを返す
        return noisy_waveform, str(noisy_path)