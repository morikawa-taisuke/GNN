import wave
import numpy as np
import math
from tqdm import tqdm
import scipy.signal as sp
from mymodule import my_func


def read_wave(in_path):
    """ ファイルの読み込み """
    wav = wave.open(in_path)
    # サンプリングレートを取る
    sampling_rate = wav.getframerate()
    # PCM形式の波形データを読み込み
    speech_signal = wav.readframes(wav.getnframes())
    # speech_signalを2バイトの数値列に変換
    speech_signal = np.frombuffer(speech_signal, dtype=np.int16)

    return speech_signal, sampling_rate

def save_wave(out_data, out_path, sampling_rate):
    # 出力先の作成
    my_func.exists_dir(dir_name=out_path)
    # 2バイトのデータに変換
    processed_data_post = out_data.astype(np.int16)
    # waveファイルに書き込む
    wave_out = wave.open(out_path, 'w')
    # モノラル:1、ステレオ:2
    wave_out.setnchannels(1)
    # サンプルサイズ2byte
    wave_out.setsampwidth(2)
    # サンプリング周波数
    wave_out.setframerate(sampling_rate)
    # データを書き込み
    wave_out.writeframes(processed_data_post)
    # ファイルを閉じる
    wave_out.close()

def pawer(wave_data):
    wave2 = wave_data * wave_data
    wave2 = np.abs(wave2)
    pawer = np.sum(wave2)
    return pawer

def pawer2(wave_data):
    print('NRR')

def nrr(clean_data, noise_data, noise_only):
    """ パワーの計算 """
    slice_clean = clean_data[:noise_only]
    slice_noise = noise_data[:noise_only]
    print(slice_clean.shape)
    clean_pawer = pawer(slice_clean)
    noise_pawer = pawer(slice_noise)+1
    print(f'clean:{clean_pawer}')
    print(f'noise:{noise_pawer}')
    """ SNの計算 """
    sn = 10 * math.log10(clean_pawer/noise_pawer)
    return sn

def nrr_evaluation(clean_path, noise_path, noise_only):
    nrr_score = nrr(clean_path, noise_path, noise_only)
    return nrr_score

def nrr_main(clean_dir, noise_dir, out_path, noise_only=30000):
    """ nrrを計算する
    #
    :param clean_path: 正解データのパス
    :param noise_path: モデル適用後データのパス
    :return nrr_score: nrr値
    """

    """ file list 取得 """
    clean_list = my_func.get_wave_filelist(clean_dir)
    noise_list = my_func.get_wave_filelist(noise_dir)

    # my_func.remove_file(file_name)
    """ 出力用のディレクトリーがない場合は作成 """
    my_func.exists_dir(out_path)
    # """ ファイルの書き込み """
    # with open(out_path, 'w') as out_file:
    #     out_file.write('clean_name,noise_name,nrr_score\n')

    for clean_file, noise_file in tqdm(zip(clean_list, noise_list)):
        """ ファイル名の取得 """
        clean_name, _ = my_func.get_fname(clean_file)
        noise_name, _ = my_func.get_fname(noise_file)
        """ データの読み込み """
        clean_data, fs = read_wave(clean_file)
        noise_data, fs = read_wave(noise_file)
        print(f'clean_data.shape:{clean_data.shape}')
        print(f'noise_data.shape:{noise_data.shape}')
        noise_data = np.zeros(len(noise_data))

        # if len(clean_data) > len(noise_data):
        #     clean_data = clean_data[:len(noise_data)]
        # elif len(clean_data) < len(noise_data):
        #     noise_data = noise_data[len(clean_data)]
        # else:
        """ nrrの計算 """
        nrr_score = nrr_evaluation(clean_path=clean_data, noise_path=noise_data, noise_only=noise_only)
        print(f'nrr_score:{nrr_score}')
        # print(f'nrr_score.dtype:{nrr_score.dtype}')

        # """ スコアの書き込み """
        # with open(out_path, 'a') as out_file:  # ファイルオープン
        #     text = f'{clean_name},{noise_name},{nrr_score}\n'
        #     out_file.write(text)  # 書き込み


if __name__ == '__main__':
    nrr_main(clean_dir='../../../sound_data/ss/mix_data/hoth_-5dB/JA01F049_hoth_-5.wav',
             noise_dir='../../../sound_data/ss/result/low2_hi5/subband/hoth_-5dB/subband_JA01F049_hoth_-5.wav',
             out_path='./nnr_noise_estimation.csv')

    nrr_main(clean_dir='../../../sound_data/ss/result/low2_hi5/subband/hoth_-5dB/subband_JA01F049_hoth_-5.wav',
             noise_dir='../../../sound_data/ss/mix_data/hoth_-5dB/JA01F049_hoth_-5.wav',
             out_path='./nnr_estimation_noise.csv')

    # a = np.array([-2,-3,-4])
    # b = np.abs(a)
    # print(np.sum(a), np.sum(b))
