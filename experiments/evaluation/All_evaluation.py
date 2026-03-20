import numpy as np
import os

from tqdm.contrib import tzip

# 閾ｪ菴懊Δ繧ｸ繝･繝ｼ繝ｫ
from evaluation.PESQ import pesq_evaluation
from evaluation.STOI import stoi_evaluation
from evaluation.SI_SDR import sisdr_evaluation
from mymodule import my_func, const


def main(target_dir, estimation_dir, out_path):
    """螳｢隕ｳ隧穂ｾ｡繧定｡後≧"""
    print("target: ", target_dir)
    print("estimation: ", estimation_dir)

    """ 蜃ｺ蜉帙ヵ繧｡繧､繝ｫ縺ｮ菴懈・"""
    my_func.make_dir(out_path)
    with open(out_path, "w") as csv_file:
        csv_file.write(f"target_dir,{target_dir}\nestimation_dir,{estimation_dir}\n")
        csv_file.write(f"{out_path}\ntarget_name,estimation_name,pesq,stoi,sisdr\n")

    """ 繝輔ぃ繧､繝ｫ繝ｪ繧ｹ繝医・菴懈・ """
    target_list = my_func.get_file_list(dir_path=target_dir, ext=".wav")
    estimation_list = my_func.get_file_list(dir_path=estimation_dir, ext=".wav")
    # print("target: ",len(target_list))
    # print("estimation: ",len(estimation_list))

    """ 蛻晄悄蛹・"""
    pesq_sum = 0
    stoi_sum = 0
    sisdr_sum = 0

    for target_file, estimation_file in tzip(target_list, estimation_list):
        """繝輔ぃ繧､繝ｫ蜷阪・蜿門ｾ・""
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)
        """ 髻ｳ貅舌・隱ｭ縺ｿ霎ｼ縺ｿ """
        target_data, _ = my_func.load_wav(target_file)
        estimation_data, _ = my_func.load_wav(estimation_file)

        # max_length = max(len(target_data), len(estimation_data))
        # target_data = np.pad(target_data, [0, max_length - len(target_data)], "constant")
        # estimation_data = np.pad(estimation_data, [0, max_length - len(estimation_data)], "constant")
        min_length = min(len(target_data), len(estimation_data))
        target_data = target_data[:min_length]
        estimation_data = estimation_data[:min_length]

        target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
        estimation_data = np.nan_to_num(estimation_data, nan=0.0, posinf=0.0, neginf=0.0)

        """ 螳｢隕ｳ隧穂ｾ｡縺ｮ險育ｮ・"""
        pesq_score = pesq_evaluation(target_data, estimation_data)
        stoi_score = stoi_evaluation(target_data, estimation_data)
        sisdr_score = sisdr_evaluation(target_data, estimation_data)
        pesq_sum += pesq_score
        stoi_sum += stoi_score
        sisdr_sum += sisdr_score

        """ 蜃ｺ蜉・繝輔ぃ繧､繝ｫ縺ｸ縺ｮ譖ｸ縺崎ｾｼ縺ｿ) """
        with open(out_path, "a") as csv_file:  # 繝輔ぃ繧､繝ｫ繧ｪ繝ｼ繝励Φ
            text = f"{target_name},{estimation_name},{pesq_score},{stoi_score},{sisdr_score}\n"  # 譖ｸ縺崎ｾｼ繧蜀・ｮｹ縺ｮ菴懈・
            csv_file.write(text)  # 譖ｸ縺崎ｾｼ縺ｿ

    """ 蟷ｳ蝮・・邂怜・(繝輔ぃ繧､繝ｫ縺ｸ縺ｮ譖ｸ縺崎ｾｼ縺ｿ) """
    pesq_ave = pesq_sum / len(estimation_list)
    stoi_ave = stoi_sum / len(estimation_list)
    sisdr_ave = sisdr_sum / len(estimation_list)
    with open(out_path, "a") as csv_file:  # 繝輔ぃ繧､繝ｫ繧ｪ繝ｼ繝励Φ
        text = f"average,,{pesq_ave},{stoi_ave},{sisdr_ave}\n"  # 譖ｸ縺崎ｾｼ繧蜀・ｮｹ縺ｮ菴懈・
        csv_file.write(text)  # 譖ｸ縺崎ｾｼ縺ｿ

    print(f"PESQ : {pesq_ave:.3f}")
    print(f"STOI : {stoi_ave:.3f}")
    print(f"SI-SDR : {sisdr_ave:.3f}")
    # print("pesq end")


if __name__ == "__main__":
    print("evaluation")

    # model_type = ["SpeqGCN", "SpeqGAT", "SpeqGCN2", "SpeqGAT2"]
    # wave_types = ["clean", "noise_only", "reverbe_only", "noise_reverb"]
    wave_types = ["noise_only"]

    # model = "subset_DEMAND_hoth_5dB_500msec"
    base_dir = "DEMAND_DEMAND"
    for wave_type in wave_types:
        # for wave_type in wave_types:
        name = f"{wave_type}"
        target_dir = f"{const.MIX_DATA_DIR}/{base_dir}/test/clean"
        estimation_dir = f"{const.MIX_DATA_DIR}/{base_dir}/test/{wave_type}"
        # target_dir = f"C:/Users/kataoka-lab/Desktop/sound_data/mix_data/GNN/subset_DEMAND_hoth_05dB_5000msec/test/clean"
        # estimation_dir = f"{const.OUTPUT_WAV_DIR}/{model}/subset_DEMAND_hoth_05dB_5000msec/{name}"
        out_csv_name = f"{base_dir}/{base_dir}/{name}.csv"
        main(target_dir=target_dir, estimation_dir=estimation_dir, out_path=os.path.join(const.EVALUATION_DIR, out_csv_name))

    # target_dir = "C:/Users/kataoka-lab/Desktop/sound_data/mix_data/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/test/clean"
    # estimation_dir = "C:/Users/kataoka-lab/Desktop/sound_data/RESULT/output_wav/UGCN/subset_DEMAND_1ch/random_node/STFT_MSE/noise_only"
    # out_csv_name = "UGCN_random_node_STFT_MSE_noise_only.csv"
    # main(target_dir=target_dir,
    #      estimation_dir=estimation_dir,
    #      out_path=os.path.join(const.EVALUATION_DIR, out_csv_name))
