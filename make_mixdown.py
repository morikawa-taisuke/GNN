"""
音声ファイルとノイズファイルを指定したSNR（信号対雑音比）で混合し、
新しい音声ファイル（ミックスダウン）を作成するスクリプト。

主な機能:
- 指定されたディレクトリから音声ファイルとノイズファイルの一覧を取得します。
- 各音声ファイルに対して、ランダムにノイズファイルを選択します。
- ノイズの長さを音声の長さに合わせてランダムにクロップ（切り出し）します。
- 音声とノイズを指定されたSNRで混合します。
- 混合後の音声ファイルを指定された出力ディレクトリに保存します。

実行方法:
スクリプトの末尾にある `if __name__ == '__main__':` ブロック内の
ディレクトリパスを環境に合わせて設定し、直接実行します。
`python make_mixdown.py`
"""

import os.path
import random

import numpy as np
import soundfile as sf
from tqdm.contrib import tzip


def random_crop(noise, target_length):
    """
    ノイズ波形を、指定された目標の長さにランダムに切り出します。

    ノイズの長さが目標長より短い場合は、目標長を超えるまでノイズを繰り返し連結してから切り出します。

    Args:
        noise (np.ndarray): 入力となるノイズの波形データ。
        target_length (int): 切り出す目標の長さ（サンプル数）。

    Returns:
        np.ndarray: 目標の長さに切り出されたノイズの波形データ。
    """
    # ノイズの長さが目標長より短い場合、ループさせて長さを確保
    if len(noise) <= target_length:
        repeat_times = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeat_times)

    # 切り出し開始位置をランダムに決定
    start = np.random.randint(0, len(noise) - target_length + 1)
    # 指定された長さでノイズを切り出して返す
    return noise[start : start + target_length]


def mix_snr(speech, noise, snr_db):
    """
    音声とノイズを、指定されたSNR(dB)になるように混合します。

    Args:
        speech (np.ndarray): 音声の波形データ。
        noise (np.ndarray): ノイズの波形データ。
        snr_db (float): 目標とする信号対雑音比 (デシベル)。

    Returns:
        np.ndarray: 音声とノイズが混合された波形データ。
    """
    # 音声とノイズのパワー（二乗平均）を計算
    speech_power = np.mean(speech**2)
    noise_power = np.mean(noise**2)

    # 目標SNRに基づき、ノイズに適用すべきゲインを計算
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    # ノイズのパワーを調整するための係数を計算し、ノイズに適用
    # noise_powerが0の場合のゼロ除算を防ぐために微小値(1e-10)を加算
    noise_gain = np.sqrt(target_noise_power / (noise_power + 1e-10))
    adjusted_noise = noise * noise_gain

    # 音声と調整後のノイズを混合して返す
    return speech + adjusted_noise


def load_wav(filepath):
    """
    WAVファイルを読み込み、波形データとサンプリングレートを返します。

    Args:
        filepath (str): 読み込むWAVファイルのパス。

    Returns:
        tuple[np.ndarray, int]: 波形データとサンプリングレートのタプル。
    """
    data, sr = sf.read(filepath)
    return data, sr


def save_wav(filepath, data, sr):
    """
    波形データをWAVファイルとして保存します。

    Args:
        filepath (str): 保存するWAVファイルのパス。
        data (np.ndarray): 保存する波形データ。
        sr (int): サンプリングレート。
    """
    sf.write(filepath, data, sr)


def get_file_list(dir_path: str, ext: str = ".wav") -> list:
    """
    指定したディレクトリ内の任意の拡張子のファイルをリストアップ

    Parameters
    ----------
    dir_path(str):ディレクトリのパス
    ext(str):拡張子

    Returns
    -------
    list[str]
    """
    if os.path.isdir(dir_path):
        return [
            f"{dir_path}/{file_path}" for file_path in os.listdir(dir_path) if os.path.splitext(file_path)[1] == ext
        ]
    else:
        return [dir_path]


def get_file_name(path: str) -> tuple:  # ->list[str, str]
    """
    パスからファイル名のみを取得

    get_file_name('./dir1/dir2/file_name.ext') -> 'file_name', 'ext'
    Parameters
    ----------
    path(str):取得するファイルのパス

    Returns
    -------
    file_name(str):ファイル名
    ext(str):拡張子
    """
    file_name, ext = os.path.splitext(os.path.basename(path))
    # print(f'file_name:{type(file_name)}')
    # print(f'ext:{type(ext)}')
    return file_name, ext


def make_dir(path: str) -> None:
    """
    目的のディレクトリを作成(ファイル名が含まれる場合,親ディレクトリを作成)

    Parameters
    ----------
    path(str):作成するディレクトリのパス

    Returns
    -------
    None
    """
    """ 作成するディレクトリが存在するかどうかを確認する """
    _, ext = os.path.splitext(path)  # dir_pathの拡張子を取得
    if len(ext) == 0:  # ディレクトリのみ場合
        os.makedirs(path, exist_ok=True)
    elif not (ext) == 0:  # ファイル名を含む場合
        os.makedirs(os.path.dirname(path), exist_ok=True)


def main(speech_dir, noise_dir, out_dir, snr=5):
    """
    音声とノイズの混合処理を実行するメイン関数。

    Args:
        speech_dir (str): 音声ファイルが格納されているディレクトリのパス。
        noise_dir (str): ノイズファイルが格納されているディレクトリのパス。
        out_dir (str): 混合後の音声ファイルを保存するディレクトリのパス。
        snr (float, optional): 混合時のSNR(dB)。デフォルトは5。
    """
    # 音声ファイルとノイズファイルのリストを取得
    speech_list = get_file_list(speech_dir)
    noise_list_source = get_file_list(noise_dir)

    # 各音声ファイルに対して、使用するノイズファイルをランダムに選択
    # これにより、毎回異なるノイズがペアになる
    noise_list_paired = [random.choice(noise_list_source) for _ in range(len(speech_list))]

    # tzipを使用してプログレスバーを表示しながらループ処理
    print(f"Mixing files for directory: {speech_dir}")
    for speech_file, noise_file in tzip(speech_list, noise_list_paired, desc=f"SNR={snr}dB Mix"):
        # 音声ファイルとノイズファイルを読み込み
        speech, sr = load_wav(speech_file)
        noise, _ = load_wav(noise_file)

        # ノイズを音声の長さに合わせてランダムにクロップ
        noise_cropped = random_crop(noise, len(speech))

        # 音声とクロップしたノイズを指定のSNRで混合
        mixed_audio = mix_snr(speech, noise_cropped, snr_db=snr)

        # 出力ファイル名を生成 (例: speech_noise_005dB.wav)
        speech_name = get_file_name(speech_file)[0]
        noise_name = get_file_name(noise_file)[0]
        out_name = f"{speech_name}_{noise_name}_{int(snr):03}dB.wav"
        out_path = os.path.join(out_dir, out_name)

        # 混合後の音声を保存
        save_wav(out_path, mixed_audio, sr)


if __name__ == "__main__":
    # 'train'と'test'の各データセットに対して処理を実行
    for train_test in ["train", "test"]:
        # 各ディレクトリのパスを設定
        speech_dir = f"sample/speech"  # 音声ファイルのディレクトリpath
        noise_dir = f"sample/noise.wav"  # ここでは単一のノイズファイルのpath
        out_dir = f"sample/noise_only"  # 出力ディレクトリのpath

        # 出力ディレクトリが存在しない場合は作成
        make_dir(out_dir)

        # メイン処理を実行
        main(speech_dir, noise_dir, out_dir, snr=5)
