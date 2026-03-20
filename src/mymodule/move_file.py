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
    繝・ぅ繝ｬ繧ｯ繝医Μ縺九ｉ莉ｻ諢上・譁・ｭ怜・繧貞性繧繝輔ぃ繧､繝ｫ蜷阪ｒ蛻･縺ｮ繝・ぅ繝ｬ繧ｯ繝医Μ縺ｫ繧ｳ繝斐・縺吶ｋ

    Parameters
    ----------
    source_dir(str):遘ｻ蜍募・縺ｮ繝・ぅ繝ｬ繧ｯ繝医Μ蜷・
    destination_dir(str):遘ｻ蜍募・縺ｮ繝・ぅ繝ｬ繧ｯ繝医Μ蜷・
    search_str(str):讀懃ｴ｢縺吶ｋ譁・ｭ怜・
    is_remove(bool):遘ｻ蜍募・縺九ｉ蜑企勁縺吶ｋ縺九←縺・° (True:蜑企勁縺吶ｋ, False:蜑企勁縺励↑縺・

    Returns
    -------
    None
    """
    """ 蜃ｺ蜉帛・縺ｮ菴懈・ """
    print("source_dir:", source_dir)
    print("destination_dir:", destination_dir)
    my_func.make_dir(destination_dir)
    """ 遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ蜀・・繝輔ぃ繧､繝ｫ繧偵Μ繧ｹ繝医い繝・・ """
    file_list = os.listdir(source_dir)

    """ 譚｡莉ｶ縺ｫ蜷郁・縺吶ｋ繝輔ぃ繧､繝ｫ繧堤ｧｻ蜍・"""
    for file in tqdm(file_list):
        if search_str in file:
            """繝代せ縺ｮ菴懈・"""
            source_file_path = os.path.join(source_dir, file)  # 遘ｻ蜍募・
            destination_file_path = os.path.join(destination_dir, file)  # 遘ｻ蜍募・
            """ 繝輔ぃ繧､繝ｫ縺ｮ繧ｳ繝斐・ """
            shutil.copy(source_file_path, destination_file_path)
            if is_remove:  # 遘ｻ蜍募・縺九ｉ蜑企勁縺吶ｋ蝣ｴ蜷・
                os.remove(source_file_path)  # 蜑企勁


def split_wav_file(source_dir: str, destination_dir: str, num_splits: int = 1) -> None:
    """
    1縺､髻ｳ貅舌ヵ繧｡繧､繝ｫ繧剃ｻｻ諢上・繝輔ぃ繧､繝ｫ縺ｫ蛻・牡縺吶ｋ(pyroomacoustics縺ｧ1ch縺ｧ骭ｲ髻ｳ縺励◆髻ｳ貅舌ｒ蛻・牡縺吶ｋ縺ｮ縺ｫ菴ｿ逕ｨ)

    Parameters
    ----------
    source_dir(str):蛻・牡縺吶ｋ蜑阪・繝・ぅ繝ｬ繧ｯ繝医Μ
    destination_dir(str):蛻・牡蠕後・繝・ぅ繝ｬ繧ｯ繝医Μ
    num_splits(int):蛻・牡謨ｰ

    Returns
    -------
    None
    """
    """ 蜃ｺ蜉帛・縺ｮ菴懈・ """
    my_func.make_dir(destination_dir)
    # 遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ蜀・・wav繝輔ぃ繧､繝ｫ繧偵Μ繧ｹ繝医い繝・・
    wav_file_list = [f for f in os.listdir(source_dir) if f.endswith(".wav")]

    for wav_file in wav_file_list:
        source_file_path = os.path.join(source_dir, wav_file)

        """隱ｭ縺ｿ霎ｼ縺ｿ"""
        with wave.open(source_file_path, "rb") as original_wav:
            """蛻・牡蠕後・繧ｵ繝ｳ繝励Ν謨ｰ繧堤ｮ怜・"""
            num_samples = original_wav.getnframes()  # 蛻・牡蜑阪・繧ｵ繝ｳ繝励Ν謨ｰ
            samples_per_split = num_samples // num_splits  # 蛻・牡蠕後・繧ｵ繝ｳ繝励Ν謨ｰ

            for i in range(num_splits):
                """蛻・牡蠕後・繝輔ぃ繧､繝ｫ蜷阪ｒ逕滓・"""
                split_file_name = f"{os.path.splitext(wav_file)[0]}_split_{i + 1}.wav"
                destination_file_path = os.path.join(destination_dir, split_file_name)
                """ 菫晏ｭ・"""
                with wave.open(destination_file_path, "wb") as split_wav:
                    split_wav.setparams(original_wav.getparams())
                    start_sample = i * samples_per_split
                    end_sample = (i + 1) * samples_per_split
                    original_wav.setpos(start_sample)
                    split_wav.writeframes(original_wav.readframes(end_sample - start_sample))


def rename_files_in_directory(directory, search_string, new_string):
    # 繝・ぅ繝ｬ繧ｯ繝医Μ蜀・・縺吶∋縺ｦ縺ｮ繝輔ぃ繧､繝ｫ繧呈､懃ｴ｢
    # directory=os.path.join(directory, "*")
    # print(directory)
    files = glob.glob(os.path.join(directory, "*"))
    print(files)

    for file in tqdm(files):
        # 繝輔ぃ繧､繝ｫ蜷阪↓讀懃ｴ｢譁・ｭ怜・縺悟性縺ｾ繧後※縺・ｋ縺九ｒ繝√ぉ繝・け
        # print(file)
        if search_string in os.path.basename(file):
            # 譁ｰ縺励＞繝輔ぃ繧､繝ｫ蜷阪ｒ逕滓・
            old_name, ext = my_func.get_file_name(file)
            # print(ext)
            old_name = f"{old_name}{ext}"
            print(old_name)
            new_file = old_name.replace(search_string, new_string)
            new_file = os.path.join(directory, new_file)
            # 繝輔ぃ繧､繝ｫ蜷阪ｒ螟画峩
            os.rename(file, new_file)
            tqdm.write(f"Renamed: {file} -> {new_file}")


"""
if __name__ == "__main__":
    # 遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ縺ｨ遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ繧呈欠螳・
    source_directory = "../../sound_data/ConvTasNet/separate/result" #"遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ縺ｮ繝代せ"
    destination_directory = "../../sound_data/ConvTasNet/separate/split" #"遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ縺ｮ繝代せ"
    # 蛻・牡謨ｰ繧呈欠螳・
    num_splits = 2
    # wav繝輔ぃ繧､繝ｫ繧貞・蜑ｲ縺励※菫晏ｭ・
    split_wav_file(source_directory, destination_directory, num_splits)
"""

if __name__ == "__main__":
    # 遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ縺ｨ遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ繧呈欠螳・
    """ 譚｡莉ｶ縺ｫ蜷郁・縺吶ｋ繝輔ぃ繧､繝ｫ縺ｮ讀懃ｴ｢譁・ｭ怜・繧呈欠螳・"""
    search_string = "p232"  # "讀懃ｴ｢譁・ｭ怜・"
    remove = True
    """ 繝・ぅ繝ｬ繧ｯ繝医Μ蜷阪・菴懈・ """
    source_directory = f"C:/Users/kataoka-lab/Desktop/sound_data/sample_data/speech/DEMAND/val"  # "遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ縺ｮ繝代せ"
    wave_type_list = [
        "clean",
        "noise_only",
        "noise_reverb",
        "reverbe_only",
    ]  # "noise_only", "noise_reverb", "reverbe_only"

    speaker_list = my_func.get_subdir_list(source_directory)
    for wave_type in speaker_list:
        destination_directory = f"{source_directory}/"  # "遘ｻ蜍募・繝・ぅ繝ｬ繧ｯ繝医Μ縺ｮ繝代せ"
        search_string = wave_type
        """ 繝輔ぃ繧､繝ｫ繧堤ｧｻ蜍・"""
        move_files(os.path.join(source_directory, wave_type), destination_directory, search_string, is_remove=remove)

    # sub_dir_list = my_func.get_subdir_list(source_directory)
    # # print(sub_dir_list)
    # for sub_dir in sub_dir_list:
    #     All_wav_list = my_func.get_wave_list(f"{source_directory}/{sub_dir}")
    #     wav_path_list = random.sample(All_wav_list, 10)
    #     my_func.make_dir(destination_directory)
    #     for wav_path in wav_path_list:
    #         """ 繝輔ぃ繧､繝ｫ縺ｮ繧ｳ繝斐・ """
    #         shutil.copy(wav_path, destination_directory)

    """ 譁・ｭ怜・縺ｮ鄂ｮ謠・"""
    # 菴ｿ逕ｨ萓・
    # C:\Users\kataoka-lab\Desktop\sound_data\dataset\subset_DEMAND_hoth_1010dB_05sec_4ch_10cm\Front\noise_only
    # directory = "C:\\Users\\kataoka-lab\\Desktop\\sound_data\\dataset\\subset_DEMAND_hoth_1010dB_05sec_4ch_10cm\\"
    # # subdir_list = my_func.get_subdir_list(directory).remove("noise_only", "")
    # # subdir_list.remove("noise_only")
    # # print(subdir_list)
    # search_string = ".npz"
    # new_name = "_{angle}.npz"
    # angle_list = ["Right", "FrontRight", "Front", "FrontLeft", "Left"]  # "Right", "FrontRight", "Front", "FrontLeft", "Left"
    # for angle in angle_list:
    #     wave_list = my_func.get_subdir_list(os.path.join(directory, angle))
    #     # wave_list = ["noise_only"]
    #     for wave_type in wave_list:
    #         print(new_name.format(angle=angle))
    #         # print(len(my_func.get_file_list(os.path.join(directory, angle, "test", wave_type))))
    #         rename_files_in_directory(os.path.join(directory, angle, wave_type), search_string, new_name.format(angle=angle))

