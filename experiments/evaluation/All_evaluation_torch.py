import numpy as np  # 蠢・ｦ√↑縺・ｴ蜷医ｂ縺ゅｊ縺ｾ縺吶′縲∝ｿｵ縺ｮ縺溘ａ谿九＠縺ｾ縺・
import os
import torch
import torchaudio

from tqdm.contrib import tzip

# 閾ｪ菴懊Δ繧ｸ繝･繝ｼ繝ｫ
from evaluation.PESQ_torch import pesq_evaluation
from evaluation.STOI_torch import stoi_evaluation
from evaluation.SISDR_torch import sisdr_evaluation
from mymodule import my_func, const


def main(target_dir, estimation_dir, out_path, device=torch.device("cpu")):
    """螳｢隕ｳ隧穂ｾ｡繧貞・縺ｦ螳溯｡後☆繧・(torchmetrics繝吶・繧ｹ縺ｮ隧穂ｾ｡髢｢謨ｰ繧剃ｽｿ逕ｨ)"""
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

    """ 蛻晄悄蛹・"""
    pesq_sum = 0
    stoi_sum = 0
    sisdr_sum = 0
    num_files = 0  # 隧穂ｾ｡蟇ｾ雎｡繝輔ぃ繧､繝ｫ縺ｮ謨ｰ繧呈ｭ｣遒ｺ縺ｫ繧ｫ繧ｦ繝ｳ繝・

    for target_file, estimation_file in tzip(target_list, estimation_list):
        """繝輔ぃ繧､繝ｫ蜷阪・蜿門ｾ・""
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)

        """ 髻ｳ貅舌・隱ｭ縺ｿ霎ｼ縺ｿ (torchaudio繧剃ｽｿ逕ｨ) """
        target_data_tensor, sr_target = torchaudio.load(target_file)
        estimation_data_tensor, sr_estimation = torchaudio.load(estimation_file)

        # 隧穂ｾ｡髢｢謨ｰ縺ｮ縺溘ａ縺ｫ蜊倅ｸ繝√Ε繝阪Ν縺ｫ螟画鋤・医ｂ縺苓､・焚繝√Ε繝阪Ν縺ｮ蝣ｴ蜷茨ｼ・
        if target_data_tensor.ndim > 1:
            target_data_tensor = target_data_tensor[0, :]
        if estimation_data_tensor.ndim > 1:
            estimation_data_tensor = estimation_data_tensor[0, :]

        # 髟ｷ縺輔・譛蟆丞､縺ｫ蜷医ｏ縺帙※繝・Φ繧ｽ繝ｫ繧偵ヨ繝ｪ繝溘Φ繧ｰ
        # 蜷・ｩ穂ｾ｡髢｢謨ｰ蜀・〒繧る聞縺戊ｪｿ謨ｴ縺瑚｡後ｏ繧後ｋ縺後√％縺薙〒荳雋ｫ縺励※陦後≧縺薙→縺ｧ蜀鈴聞諤ｧ繧呈ｸ帙ｉ縺帙ｋ
        min_length = min(target_data_tensor.shape[0], estimation_data_tensor.shape[0])
        target_data_tensor = target_data_tensor[:min_length]
        estimation_data_tensor = estimation_data_tensor[:min_length]

        # NaN/Inf蛟､繧・.0縺ｫ鄂ｮ縺肴鋤縺茨ｼ・yTorch繧剃ｽｿ逕ｨ・・
        target_data_tensor = torch.nan_to_num(target_data_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        estimation_data_tensor = torch.nan_to_num(estimation_data_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        """ 螳｢隕ｳ隧穂ｾ｡縺ｮ險育ｮ・(torchmetrics繝吶・繧ｹ縺ｮ髢｢謨ｰ繧貞他縺ｳ蜃ｺ縺・ """
        # 蜷・ｩ穂ｾ｡髢｢謨ｰ縺ｯPyTorch繝・Φ繧ｽ繝ｫ繧貞女縺大叙繧翫∝腰荳縺ｮfloat蛟､繧定ｿ斐＠縺ｾ縺・
        pesq_score = pesq_evaluation(target_data_tensor, estimation_data_tensor, device=device)
        stoi_score = stoi_evaluation(target_data_tensor, estimation_data_tensor, device=device)
        sisdr_score = sisdr_evaluation(target_data_tensor, estimation_data_tensor, device=device)

        pesq_sum += pesq_score
        stoi_sum += stoi_score
        sisdr_sum += sisdr_score
        num_files += 1

        """ 蜃ｺ蜉・繝輔ぃ繧､繝ｫ縺ｸ縺ｮ譖ｸ縺崎ｾｼ縺ｿ) """
        with open(out_path, "a") as csv_file:
            text = f"{target_name},{estimation_name},{pesq_score:.4f},{stoi_score:.4f},{sisdr_score:.4f}\n"  # 繝輔か繝ｼ繝槭ャ繝医ｒ隱ｿ謨ｴ
            csv_file.write(text)

    """ 蟷ｳ蝮・・邂怜・(繝輔ぃ繧､繝ｫ縺ｸ縺ｮ譖ｸ縺崎ｾｼ縺ｿ) """
    if num_files > 0:
        pesq_ave = pesq_sum / num_files
        stoi_ave = stoi_sum / num_files
        sisdr_ave = sisdr_sum / num_files
    else:
        pesq_ave = stoi_ave = sisdr_ave = 0.0  # 繝輔ぃ繧､繝ｫ縺後↑縺・ｴ蜷・

    with open(out_path, "a") as csv_file:
        text = f"average,,{pesq_ave:.4f},{stoi_ave:.4f},{sisdr_ave:.4f}\n"  # 繝輔か繝ｼ繝槭ャ繝医ｒ隱ｿ謨ｴ
        csv_file.write(text)

    print(f"PESQ : {pesq_ave:.3f}")
    print(f"STOI : {stoi_ave:.3f}")
    print(f"SI-SDR : {sisdr_ave:.3f}")


if __name__ == "__main__":
    print("evaluation")

    wave_types = ["clean", "noise_only", "reverbe_only", "noise_reverb"]

    model = "subset_DEMAND_hoth_5dB_500msec"
    for wave_type in wave_types:
        name = f"{model}_{wave_type}"
        target_dir = f"C:/Users/kataoka-lab/Desktop/sound_data/mix_data/GNN/subset_DEMAND_hoth_5dB_500msec/test/clean"
        estimation_dir = f"C:/Users/kataoka-lab/Desktop/sound_data/mix_data/GNN/subset_DEMAND_hoth_5dB_500msec/test/{wave_type}"
        out_csv_name = f"subset_DEMAND_hoth_5dB_500msec/{name}.csv"
        main(target_dir=target_dir, estimation_dir=estimation_dir, out_path=os.path.join(const.EVALUATION_DIR, out_csv_name))
