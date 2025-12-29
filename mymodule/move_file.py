# coding:utf-8

import glob
import os
import shutil
import wave

# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tqdm import tqdm

import my_func


def move_files(source_dir: str, destination_dir: str, search_str: str, is_remove: bool = False) -> None:
    """
    ディレクトリから任意の文字列を含むファイル名を別のディレクトリにコピーする

    Parameters
    ----------
    source_dir(str):移動元のディレクトリ名
    destination_dir(str):移動先のディレクトリ名
    search_str(str):検索する文字列
    is_remove(bool):移動元から削除するかどうか (True:削除する, False:削除しない)

    Returns
    -------
    None
    """
    """ 出力先の作成 """
    print("source_dir:", source_dir)
    print("destination_dir:", destination_dir)
    my_func.make_dir(destination_dir)
    """ 移動元ディレクトリ内のファイルをリストアップ """
    file_list = os.listdir(source_dir)

    """ 条件に合致するファイルを移動 """
    for file in tqdm(file_list):
        if search_str in file:
            """パスの作成"""
            source_file_path = os.path.join(source_dir, file)  # 移動元
            destination_file_path = os.path.join(destination_dir, file)  # 移動先
            """ ファイルのコピー """
            shutil.copy(source_file_path, destination_file_path)
            if is_remove:  # 移動元から削除する場合
                os.remove(source_file_path)  # 削除


def split_wav_file(source_dir: str, destination_dir: str, num_splits: int = 1) -> None:
    """
    1つ音源ファイルを任意のファイルに分割する(pyroomacousticsで1chで録音した音源を分割するのに使用)

    Parameters
    ----------
    source_dir(str):分割する前のディレクトリ
    destination_dir(str):分割後のディレクトリ
    num_splits(int):分割数

    Returns
    -------
    None
    """
    """ 出力先の作成 """
    my_func.make_dir(destination_dir)
    # 移動元ディレクトリ内のwavファイルをリストアップ
    wav_file_list = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

    for wav_file in wav_file_list:
        source_file_path = os.path.join(source_dir, wav_file)

        """読み込み"""
        with wave.open(source_file_path, "rb") as original_wav:
            """分割後のサンプル数を算出"""
            num_samples = original_wav.getnframes()  # 分割前のサンプル数
            samples_per_split = num_samples // num_splits  # 分割後のサンプル数

            for i in range(num_splits):
                """分割後のファイル名を生成"""
                split_file_name = f"{os.path.splitext(wav_file)[0]}_split_{i + 1}.wav"
                destination_file_path = os.path.join(destination_dir, split_file_name)
                """ 保存 """
                with wave.open(destination_file_path, "wb") as split_wav:
                    split_wav.setparams(original_wav.getparams())
                    start_sample = i * samples_per_split
                    end_sample = (i + 1) * samples_per_split
                    original_wav.setpos(start_sample)
                    split_wav.writeframes(original_wav.readframes(end_sample - start_sample))


def rename_files_in_directory(directory, search_string, new_string):
    # ディレクトリ内のすべてのファイルを検索
    # directory=os.path.join(directory, "*")
    # print(directory)
    files = glob.glob(os.path.join(directory, "*"))
    print(files)

    for file in tqdm(files):
        # ファイル名に検索文字列が含まれているかをチェック
        # print(file)
        if search_string in os.path.basename(file):
            # 新しいファイル名を生成
            old_name, ext = my_func.get_file_name(file)
            # print(ext)
            old_name = f"{old_name}{ext}"
            print(old_name)
            new_file = old_name.replace(search_string, new_string)
            new_file = os.path.join(directory, new_file)
            # ファイル名を変更
            os.rename(file, new_file)
            tqdm.write(f"Renamed: {file} -> {new_file}")


"""
if __name__ == "__main__":
    # 移動元ディレクトリと移動先ディレクトリを指定
    source_directory = "../../sound_data/ConvTasNet/separate/result" #"移動元ディレクトリのパス"
    destination_directory = "../../sound_data/ConvTasNet/separate/split" #"移動先ディレクトリのパス"
    # 分割数を指定
    num_splits = 2
    # wavファイルを分割して保存
    split_wav_file(source_directory, destination_directory, num_splits)
"""

if __name__ == "__main__":
    # 移動元ディレクトリと移動先ディレクトリを指定
    """ 条件に合致するファイルの検索文字列を指定 """
    search_string = "p232"  # "検索文字列"
    remove = True
    """ ディレクトリ名の作成 """
    source_directory = f"C:/Users/kataoka-lab/Desktop/sound_data/sample_data/speech/DEMAND/val"  # "移動元ディレクトリのパス"
    wave_type_list = [
        "clean",
        "noise_only",
        "noise_reverbe",
        "reverbe_only",
    ]  # "noise_only", "noise_reverbe", "reverbe_only"

    speeker_list = my_func.get_subdir_list(source_directory)
    """ ファイルを移動 """
    # for wave_type in speeker_list:
    #     destination_directory = f"{source_directory}/"  # "移動先ディレクトリのパス"
    #     search_string = wave_type
    #     move_files(os.path.join(source_directory, wave_type), destination_directory, search_string, is_remove=remove)

    # sub_dir_list = my_func.get_subdir_list(source_directory)
    # # print(sub_dir_list)
    # for sub_dir in sub_dir_list:
    #     All_wav_list = my_func.get_wave_list(f"{source_directory}/{sub_dir}")
    #     wav_path_list = random.sample(All_wav_list, 10)
    #     my_func.make_dir(destination_directory)
    #     for wav_path in wav_path_list:
    #         """ ファイルのコピー """
    #         shutil.copy(wav_path, destination_directory)

    """ 文字列の置換 """
    # 使用例
    #     C:/Users/kataoka-lab/Desktop/sound_data/dataset/subset_DEMAND_hoth_1010dB_05sec_4ch_10cm/Front/noise_only
    directory = "C:/Users/kataoka-lab/Desktop/PyRoomAcoustics/scripts/precomputed_params"
    # subdir_list = my_func.get_subdir_list(directory).remove("noise_only", "")
    # subdir_list.remove("noise_only")
    # print(subdir_list)
    search_string = ".0"
    new_name = ""
    # angle_list = ["Right", "FrontRight", "Front", "FrontLeft", "Left"]  # "Right", "FrontRight", "Front", "FrontLeft", "Left"
    # for angle in angle_list:
    # wave_list = my_func.get_subdir_list(os.path.join(directory, angle))
    # wave_list = ["noise_only"]
    # for wave_type in wave_list:
    #     print(new_name.format(angle=angle))
        # print(len(my_func.get_file_list(os.path.join(directory, angle, "test", wave_type))))
    rename_files_in_directory(directory, search_string, new_name)

