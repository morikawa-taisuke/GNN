import os
import soundfile as sf
import numpy as np
from builtins import print
from tqdm import tqdm



def list_audio_durations_in_directory(directory_path: str):
    """
    指定されたディレクトリ内のすべての音声ファイルの音声長をリストアップして出力します。

    Args:
        directory_path (str): スキャンするディレクトリのパス。
    """
    # soundfileが一般的に扱える拡張子 (必要に応じて追加・変更してください)
    supported_extensions = ('.wav', '.flac', '.ogg', '.mp3', '.aiff', '.au')

    if not os.path.isdir(directory_path):
        print(f"エラー: ディレクトリ '{directory_path}' が見つかりません。")
        return

    print(f"\nディレクトリ '{directory_path}' 内の音声ファイルをスキャンしています...\n")
    found_audio_files = False
    output_lines = []
    # output_lines = np.array([])

    for filename in tqdm(os.listdir(directory_path)):
        filepath = os.path.join(directory_path, filename)

        # 音声ファイルの情報を取得
        info = sf.info(filepath)
        frames = info.frames  # サンプル数
        samplerate = info.samplerate  # サンプリングレート (Hz)

        output_lines.append(frames)
        # print(frames)
        # np.append(output_lines, frames)

    print(max(output_lines))



if __name__ == "__main__":
    target_dir = "C:/Users/kataoka-lab/Desktop/sound_data/mix_data/DEMAND_1ch/condition_1/train/noise_reverbe"
    # target_dir = input("音声ファイルが含まれるディレクトリのパスを入力してください: ")
    list_audio_durations_in_directory(target_dir)