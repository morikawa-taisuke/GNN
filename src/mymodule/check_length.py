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
    リスト�EリストをCSVファイルに保存します、E

    Args:
        data_list (list): 保存したいチE�Eタを含むリスト�Eリスト（二次允E��スト）、E
                          吁E�E部リスト�ECSVの1行に対応します、E
        filename (str): 保存するCSVファイルのパスとファイル名（侁E 'output.csv'�E�、E
        header (list, optional): CSVファイルのヘッダー行として使用する斁E���Eのリスト、E
                                 持E��しなぁE��合�Eヘッダーは書き込まれません、E
    """
    try:
        my_func.make_dir(filename)
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            if header:
                csv_writer.writerow(header)  # ヘッダーを書き込む

            csv_writer.writerows(data_list)  # チE�Eタ行を書き込む
        # print(f"'{filename}' にチE�Eタを保存しました、E)
    except IOError as e:
        print(f"ファイル '{filename}' への書き込み中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

# npzファイルの読み込み
def load_dataset(dataset_path:str, out_dir:str):
    """
    npzファイルから入力データと教師チE�Eタを読み込む

    Parameters
    ----------
    dataset_path(str):チE�EタセチE��のパス

    Returns
    -------
    mix_list:入力信号
    target_list:目皁E��号
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
    dataset_path = f"{const.DATASET_DIR}/DEMAND_1ch/condition_4/noise_reverb"
    out_path = f"{const.DATASET_DIR}/DEMAND_1ch/condition_4/edge_idx/"
    edge_idx = load_dataset(dataset_path, out_path)
    # print(len(edge_idx))
    # for edge in edge_idx:
    #     print(np.array(edge).shape)

if __name__ == '__main__':
    main()