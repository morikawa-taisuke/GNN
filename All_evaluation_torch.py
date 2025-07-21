import numpy as np  # 必要ない場合もありますが、念のため残します
import os
import torch
import torchaudio

from tqdm.contrib import tzip

# 自作モジュール
from evaluation.PESQ_torch import pesq_evaluation
from evaluation.STOI_torch import stoi_evaluation
from evaluation.SISDR_torch import sisdr_evaluation
from mymodule import my_func, const


def main(target_dir, estimation_dir, out_path, device=torch.device("cpu")):
    """客観評価を全て実行する (torchmetricsベースの評価関数を使用)"""
    print("target: ", target_dir)
    print("estimation: ", estimation_dir)

    """ 出力ファイルの作成"""
    my_func.make_dir(out_path)
    with open(out_path, "w") as csv_file:
        csv_file.write(f"target_dir,{target_dir}\nestimation_dir,{estimation_dir}\n")
        csv_file.write(f"{out_path}\ntarget_name,estimation_name,pesq,stoi,sisdr\n")

    """ ファイルリストの作成 """
    target_list = my_func.get_file_list(dir_path=target_dir, ext=".wav")
    estimation_list = my_func.get_file_list(dir_path=estimation_dir, ext=".wav")

    """ 初期化 """
    pesq_sum = 0
    stoi_sum = 0
    sisdr_sum = 0
    num_files = 0  # 評価対象ファイルの数を正確にカウント

    for target_file, estimation_file in tzip(target_list, estimation_list):
        """ファイル名の取得"""
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)

        """ 音源の読み込み (torchaudioを使用) """
        target_data_tensor, sr_target = torchaudio.load(target_file)
        estimation_data_tensor, sr_estimation = torchaudio.load(estimation_file)

        # 評価関数のために単一チャネルに変換（もし複数チャネルの場合）
        if target_data_tensor.ndim > 1:
            target_data_tensor = target_data_tensor[0, :]
        if estimation_data_tensor.ndim > 1:
            estimation_data_tensor = estimation_data_tensor[0, :]

        # 長さの最小値に合わせてテンソルをトリミング
        # 各評価関数内でも長さ調整が行われるが、ここで一貫して行うことで冗長性を減らせる
        min_length = min(target_data_tensor.shape[0], estimation_data_tensor.shape[0])
        target_data_tensor = target_data_tensor[:min_length]
        estimation_data_tensor = estimation_data_tensor[:min_length]

        # NaN/Inf値を0.0に置き換え（PyTorchを使用）
        target_data_tensor = torch.nan_to_num(target_data_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        estimation_data_tensor = torch.nan_to_num(estimation_data_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        """ 客観評価の計算 (torchmetricsベースの関数を呼び出し) """
        # 各評価関数はPyTorchテンソルを受け取り、単一のfloat値を返します
        pesq_score = pesq_evaluation(target_data_tensor, estimation_data_tensor, device=device)
        stoi_score = stoi_evaluation(target_data_tensor, estimation_data_tensor, device=device)
        sisdr_score = sisdr_evaluation(target_data_tensor, estimation_data_tensor, device=device)

        pesq_sum += pesq_score
        stoi_sum += stoi_score
        sisdr_sum += sisdr_score
        num_files += 1

        """ 出力(ファイルへの書き込み) """
        with open(out_path, "a") as csv_file:
            text = f"{target_name},{estimation_name},{pesq_score:.4f},{stoi_score:.4f},{sisdr_score:.4f}\n"  # フォーマットを調整
            csv_file.write(text)

    """ 平均の算出(ファイルへの書き込み) """
    if num_files > 0:
        pesq_ave = pesq_sum / num_files
        stoi_ave = stoi_sum / num_files
        sisdr_ave = sisdr_sum / num_files
    else:
        pesq_ave = stoi_ave = sisdr_ave = 0.0  # ファイルがない場合

    with open(out_path, "a") as csv_file:
        text = f"average,,{pesq_ave:.4f},{stoi_ave:.4f},{sisdr_ave:.4f}\n"  # フォーマットを調整
        csv_file.write(text)

    print(f"PESQ : {pesq_ave:.3f}")
    print(f"STOI : {stoi_ave:.3f}")
    print(f"SI-SDR : {sisdr_ave:.3f}")


if __name__ == "__main__":
    print("evaluation")

    wave_types = ["clean", "noise_only", "reverbe_only", "noise_reverbe"]

    model = "subset_DEMAND_hoth_5dB_500msec"
    for wave_type in wave_types:
        name = f"{model}_{wave_type}"
        target_dir = f"C:/Users/kataoka-lab/Desktop/sound_data/mix_data/GNN/subset_DEMAND_hoth_5dB_500msec/test/clean"
        estimation_dir = f"C:/Users/kataoka-lab/Desktop/sound_data/mix_data/GNN/subset_DEMAND_hoth_5dB_500msec/test/{wave_type}"
        out_csv_name = f"subset_DEMAND_hoth_5dB_500msec/{name}.csv"
        main(target_dir=target_dir, estimation_dir=estimation_dir, out_path=os.path.join(const.EVALUATION_DIR, out_csv_name))
