import numpy as np
import os

from tqdm.contrib import tzip
# 自作モジュール
from evaluation.PESQ import pesq_evaluation
from evaluation.STOI import stoi_evaluation
from evaluation.SI_SDR import sisdr_evaluation
from mymodule import my_func, const


def main(target_dir, estimation_dir, out_path):
    """ 客観評価を行う


    """
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
    # print("target: ",len(target_list))
    # print("estimation: ",len(estimation_list))

    """ 初期化 """
    pesq_sum = 0
    stoi_sum = 0
    sisdr_sum = 0

    for target_file, estimation_file in tzip(target_list, estimation_list):
        """ ファイル名の取得 """
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)
        """ 音源の読み込み """
        target_data, _ = my_func.load_wav(target_file)
        estimation_data, _ = my_func.load_wav(estimation_file)

        max_length = max(len(target_data), len(estimation_data))
        target_data = np.pad(target_data, [0, max_length - len(target_data)], "constant")
        estimation_data = np.pad(estimation_data, [0, max_length - len(estimation_data)], "constant")

        target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
        estimation_data = np.nan_to_num(estimation_data, nan=0.0, posinf=0.0, neginf=0.0)

        print("target: ", target_data)
        print("estimation: ", estimation_data)
        print("target: ", target_data.shape)
        print("estimation: ", estimation_data.shape)

        """ 客観評価の計算 """
        pesq_score = pesq_evaluation(target_data, estimation_data)
        stoi_score = stoi_evaluation(target_data, estimation_data)
        sisdr_score = sisdr_evaluation(target_data, estimation_data)
        pesq_sum += pesq_score
        stoi_sum += stoi_score
        sisdr_sum += sisdr_score

        """ 出力(ファイルへの書き込み) """
        with open(out_path, "a") as csv_file:  # ファイルオープン
            text = f"{target_name},{estimation_name},{pesq_score},{stoi_score},{sisdr_score}\n"  # 書き込む内容の作成
            csv_file.write(text)  # 書き込み

    """ 平均の算出(ファイルへの書き込み) """
    pesq_ave=pesq_sum/len(estimation_list)
    stoi_ave=stoi_sum/len(estimation_list)
    sisdr_ave=sisdr_sum/len(estimation_list)
    with open(out_path, "a") as csv_file:  # ファイルオープン
        text = f"average,,{pesq_ave},{stoi_ave},{sisdr_ave}\n"  # 書き込む内容の作成
        csv_file.write(text)  # 書き込み

    print(f"PESQ : {pesq_ave}")
    print(f"STOI : {stoi_ave}")
    print(f"SI-SDR : {sisdr_ave}")
    # print("pesq end")

if __name__ == "__main__":
    print("evaluation")
    main(target_dir=os.path.join(const.MIX_DATA_DIR, "JA_hoth_00dB",  "test", "target"),
         estimation_dir=os.path.join(const.OUTPUT_WAV_DIR, "JA_hoth_00dB"),
         out_path=os.path.join(const.EVALUATION_DIR, "JA_hoth_00dB.csv"))
