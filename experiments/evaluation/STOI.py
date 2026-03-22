# coding:utf-8

import mymodule.const
from tqdm import tqdm
from mymodule import my_func, const
import soundfile as sf
import pystoi


def stoi_evaluation(target_data, estimation_data):
    """stoi値の算出

    :param target_data: 正解データのデータ
    :param estimation_data: モデル適用後データのデータ
    :return stoi_score: stoi値
    """

    fs = mymodule.const.SR

    """ stoiの計算 """
    # pystoi.stoi(正解データ, 推測したデータ, サンプリングレート(デフォルト:16kHz))
    stoi_score = pystoi.stoi(target_data, estimation_data, fs, extended=False)

    return stoi_score


"""
clean:
noise:
file_name:
"""
def stoi_main(target_dir, estimation_dir, out_path):
    """ file list 取得 """
    target_list = my_func.get_file_list(target_dir)
    estimation_list = my_func.get_file_list(estimation_dir)

    # my_func.remove_file(file_name)
    """ 出力用のディレクトリーがない場合は作成 """
    my_func.make_dir(out_path)
    # print(f'out_path:{out_path}')
    with open(out_path, 'w') as file:
        file.write('target_name,estimation_name,stoi_score\n')
    SR = const.SR
    for target_file, estimation_file in tqdm(zip(target_list,estimation_list)):
        """ ファイル名の取得 """
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_list(estimation_file)
        """ データの読み込み """
        target_data, fs = sf.read(target_file)
        estimation_data, fs = sf.read(estimation_file)


        stoi_score = stoi_evaluation(target_data, estimation_data)
        #print("d", d.dtype)
        #with open(file_name, 'a') as f:
        with open(out_path, 'a') as out_file:
            # print("{:.10f}".format(Decimal(stoi_score)), file=f)
            text = f'{target_name},{estimation_name},{stoi_score}\n'    # 書き込む内容の作成
            out_file.write(text)                                        # 書き込み
    

# mix_reverb_snr30_unet-dnn-wpe_LSTM_mix-target
if __name__ == '__main__':
  """
  stoi_main('../../data_sample/test/tasnet_mix_re_03-k_2030-re_0-pre005',
            '../../data_sample/test/clean_0.5_test',
            '../../data_sample/test/stoi-tasnet_mix_re_03-k_2030-re_0-pre005.csv')
  """
  #"""
  stoi_main('../../sound_data/ConvtasNet/test/0dB/target',
            '../../sound_data/ConvtasNet/test/0dB/test',
            '../../sound_data/ConvtasNet/evaluation/stoi_CMU_0dB_before_spectorogram.csv')

  stoi_main('../../sound_data/ConvtasNet/test/0dB/target',
            '../../sound_data/ConvtasNet/result/0dB/test',
            '../../sound_data/ConvtasNet/evaluation/stoi_CMU_0dB_after_spectorogram.csv')
