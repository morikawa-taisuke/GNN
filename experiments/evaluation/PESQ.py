import pesq
from tqdm import tqdm
import os
import sys
sys.path.append('C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\mymodule\\')
from mymodule import my_func


def pesq_evaluation(target_data, estimation_data):
    """pesq値の算出

    :param target_data: 正解データのデータ
    :param estimation_data: モデル適用後データのデータ
    :return pesq_score: pesq値
    """

    """ pesqの計算 """
    # pesq.pesq(正解データ, 推測したデータ, サンプリングレート(デフォルト:16kHz))
    pesq_score = pesq.pesq(fs=16000, ref=target_data, deg=estimation_data)  # pesqの計算

    return pesq_score


def pesq_main(target_dir, estimation_dir, out_path):
    """pesqの算出とcsvファイルへの書き込み

    :param target_dir: 正解データのディレクトリ
    :param estimation_dir: モデル適用後データのディレクトリ
    :param dataset_dir:
    """
    # print('pesq start')

    """ 出力ファイルの作成"""
    my_func.make_dir(out_path)
    with open(out_path, 'w') as out_file:
        out_file.write('target_name,estimation_name,pesq_score\n')

    """ ファイルリストの作成 """
    target_list = my_func.get_file_list(target_dir)
    estimation_list = my_func.get_file_list(estimation_dir)

    for target_file, estimation_file in tqdm(zip(target_list, estimation_list)):
        """ ファイル名の取得 """
        target_name,_ = my_func.get_file_name(target_file)
        estimation_name,_ = my_func.get_file_name(estimation_file)
        """ 音源の読み込み """
        target_data, _ = my_func.load_wav(target_file)
        estimation_data, _ = my_func.load_wav(estimation_file)
        """ pesq値の計算 """
        pesq_score = pesq_evaluation(target_data, estimation_data)

        """ 出力(ファイルへの書き込み) """
        with open(out_path, 'a') as out_file:                           # ファイルオープン
            text = f'{target_name},{estimation_name},{pesq_score}\n'    # 書き込む内容の作成
            out_file.write(text)                                        # 書き込み

    # print('pesq end')


if __name__ == '__main__':

  # original_path = "正解データ(clean)のパス"     # 正解データ
  # target_path = "モデルを適用したパス" # モデルデータ

  target_dir = '../../sound_data/LSTM/mix_data/JA_hoth_10db_05sec/test' # 正解データ
  estimation_dir = '../../sound_data/UNet/result/JA_hoth_10db_05sec'     # モデルデータ
  out_name = 'JA_hoth_10db_05sec'

  pesq_main(target_dir=target_dir,
            estimation_dir=estimation_dir,
            out_path=f'pesq1/{out_name}.csv')
  # original_list = my_func.get_wave_filelist(original_path)
  # target_list = my_func.get_wave_filelist(target_path)
  # for original_file, target_file in zip(original_list, target_list):
  #   # print(f'original_file:{original_file}')
  #   # print(f'target_file:{target_file}')
  #   pesq_evaluation(original_file, target_file)
