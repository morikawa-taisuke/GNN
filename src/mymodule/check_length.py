import numpy as np
from librosa.util import find_files
import torch
from networkx.classes import edges
from sympy.printing.numpy import const
from torch.utils.data import DataLoader
import torchaudio
import csv
from tqdm import tqdm
import os

from mymodule import my_func, const


def save_list_to_csv(data_list, filename, header=None):
    """
    リストのリストをCSVファイルに保存します。

    Args:
        data_list (list): 保存したいデータを含むリストのリスト（二次元リスト）。
                          各内部リストはCSVの1行に対応します。
        filename (str): 保存するCSVファイルのパスとファイル名（例: 'output.csv'）。
        header (list, optional): CSVファイルのヘッダー行として使用する文字列のリスト。
                                 指定しない場合はヘッダーは書き込まれません。
    """
    try:
        my_func.make_dir(filename)
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            if header:
                csv_writer.writerow(header)  # ヘッダーを書き込む

            csv_writer.writerows(data_list)  # データ行を書き込む
        # print(f"'{filename}' にデータを保存しました。")
    except IOError as e:
        print(f"ファイル '{filename}' への書き込み中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

# npzファイルの読み込み
def load_dataset(dataset_path:str, out_dir:str):
    """
    npzファイルから入力データと教師データを読み込む

    Parameters
    ----------
    dataset_path(str):データセットのパス

    Returns
    -------
    mix_list:入力信号
    target_list:目的信号
    """
    # print('\nload_dataset')
    dataset_list = find_files(dataset_path, ext="npz", case_sensitive=True)
    # print('dataset_list:', len(dataset_list))
    for dataset_file in tqdm(dataset_list):
        dat = np.load(dataset_file)  # datファイルの読み込み
        # print(f'dat:{dat.files}')
        # print('dat:', dat['target'])
        # mix_list.append(dat[const.DATASET_KEY_MIXDOWN])  # 入力データの追加
        # print(np.array(dat['edge_index']).shape)
        data = dat['edge_index']
        save_list_to_csv(data_list=data, filename=os.path.join(out_dir, my_func.get_file_name(dataset_file)[0] + ".csv"))
    # print('load:np.array(mix_list.shape):', np.array(mix_list).shape)
    # print('load:np.array(target_list.shape):', np.array(target_list).shape)
    # print('load_dataset\n')


def main():
    print("main")
    dataset_path = f"{const.DATASET_DIR}/DEMAND_1ch/condition_4/noise_reverbe"
    out_path = f"{const.DATASET_DIR}/DEMAND_1ch/condition_4/edge_idx/"
    edge_idx = load_dataset(dataset_path, out_path)
    # print(len(edge_idx))
    # for edge in edge_idx:
    #     print(np.array(edge).shape)

if __name__ == '__main__':
    main()