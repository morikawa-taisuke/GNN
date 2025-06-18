# coding:utf-8
""" データセットを作成する """
import os
from typing import Any

import numpy as np
from librosa.core import stft
import scipy.signal as sp
import torch
import torchaudio
import scipy.signal
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from tqdm.contrib import tzip
from tqdm import tqdm
import csv
import pandas as pd
import time

# 自作モジュール
from mymodule import const, my_func


def preEmphasis(signal, p):
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, signal)

def save_multi_stft(PATH, PATH2, PATH_stft):

    #PATH_speech_wav = PATH + "target"
    """
    for i in range(4):
        if i == 0:
            RAD = "010"
        elif i == 1:
            RAD = "030"
        elif i == 2:
            RAD = "050"
        elif i == 3:
            RAD = "070"
    """

    # 源音 と 雑音入り音声 のファイルのリストを作る
    global mix_multi_tensor
    filelist_mixdown = my_func.get_file_list(PATH, ext=".wav")
    filelist_speech = my_func.get_file_list(PATH2, ext=".wav")
    #filelist_reverb = my_func.get_wave_filelist(PATH3)
    print("ORIGINAL_number of noise file", len(filelist_mixdown))
    print("MVDR_number of speech file", len(filelist_speech))
    #print("MVDR_number of reverb file", len(filelist_reverb))

    #print("filelist", filelist_mixdown)

    # 出力用のディレクトリーがない場合は作成
    my_func.make_dir(PATH_stft)

    #scaler = StandardScaler()
    #scaler = MinMaxScaler()

    mix = np.zeros([0, 128000])
    clean = np.zeros([0, 128000])

    variation = int(len(filelist_mixdown) / 2) #384
    #print("variation", variation)
    audio = int(variation / 4) #96
    #print("audio", audio)

    for fmixdown in filelist_mixdown:
        #print("fmixdown", fmixdown)
        y_mixdown, prm = my_func.load_wav(fmixdown)
        y_mixdown = y_mixdown.astype(np.float32)
        f, t, spectrogram_mix = sp.stft(y_mixdown, fs=16000, window="hann", nperseg=512, noverlap=512 - 128)
        mix_multi_tensor = torch.from_numpy(y_mixdown)
        spectrogram_mix_torch = torch.stft(mix_multi_tensor, n_fft=512, hop_length=128, window=torch.hann_window(512),return_complex=True)
        #print("spectrogram_mix_torch", spectrogram_mix_torch.shape)
        #spectrogram_mix_torch = torch.reshape(spectrogram_mix_torch,
        #                                   (spectrogram_mix_torch.size(dim=2), spectrogram_mix_torch.size(dim=0), spectrogram_mix_torch.size(dim=1)))
        print("spectrogram_mix", spectrogram_mix.shape)
        print(spectrogram_mix * 1025)
        print("spectrogram_mix_torch", spectrogram_mix_torch.shape)
        print(spectrogram_mix_torch)

        mix_length = len(y_mixdown)
        # max_reverb = len(y_reverb)

        if mix_length > 128000:
            y_mixdown = y_mixdown[:128000]

        y_mixdown = y_mixdown[np.newaxis, :]
        mix = np.r_[mix, y_mixdown]

    print("noise_reverberation", mix.shape)

    for fspeech in filelist_speech:
        #print("fspeech", fspeech)
        y_specch, prm = my_func.load_wav(fspeech)
        y_specch = y_specch.astype(np.float32)

        max_specch = len(y_specch)
        #max_reverb = len(y_reverb)

        if max_specch > 128000:
            y_specch = y_specch[:128000]

        y_specch = y_specch[np.newaxis, :]
        clean = np.r_[clean, y_specch]

    f, t, spectrogram_mix_torch = torch.stft(mix_multi_tensor, n_fft=512, hop_length=128)  # , window=torch.hann_window())

    print("clean", clean.shape)

    for i in range(variation):
        #print("i", i)
        mix_multi = np.zeros([0, 128000])
        for j in range(2):
            y_mix = mix[i+i+j, :]
            y_mix = y_mix[np.newaxis, :]
            #print("y_mix", y_mix.shape)
            mix_multi = np.r_[mix_multi, y_mix]

        mix_multi = mix_multi.astype(np.float32)
        #print("mix_multi", mix_multi.shape)
        f, t, spectrogram_mix = sp.stft(mix_multi, fs=16000, window="hann", nperseg=512, noverlap=512 - 128)
        mix_multi_tensor = torch.from_numpy(mix_multi)
        spectrogram_mix_torch = torch.stft(mix_multi_tensor, n_fft=512,hop_length=128, window=torch.hann_window(512))
        spectrogram_mix_torch = torch.reshape(spectrogram_mix_torch, (spectrogram_mix_torch[2], spectrogram_mix_torch[0], spectrogram_mix_torch[1]))
        print("spectrogram_mix", spectrogram_mix.shape)
        print(spectrogram_mix)
        print("spectrogram_mix_torch", spectrogram_mix_torch.shape)
        print(spectrogram_mix_torch)

        #spectrogram_mix = (np.abs(stft(y_mixdown, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE))).astype(np.float32)

        spectrogram_mix = spectrogram_mix.real ** 2 + spectrogram_mix.imag ** 2
        spectrogram_mix = np.maximum(spectrogram_mix, 1.e-8)
        spectrogram_mix = np.log(spectrogram_mix)

        clean = clean.astype(np.float32)
        f, t, spectrogram_target = sp.stft(clean[i], fs=16000, window="hann", nperseg=512, noverlap=512 - 128)
        #print("spectrogram_target", spectrogram_target.shape)
        spectrogram_target = spectrogram_target.real ** 2 + spectrogram_target.imag ** 2
        spectrogram_target = np.maximum(spectrogram_target, 1.e-8)
        spectrogram_target = np.log(spectrogram_target)

        # スペクトルを npzファイルとして保存する
        #path_fmixdown, _ = os.path.splitext(os.path.basename(filelist_mixdown[(i+1)*j]))
        path_fspeech, _ = os.path.splitext(os.path.basename(filelist_speech[i]))
        #path_freverb, _ = os.path.splitext(os.path.basename(freverb))

        # スペクトルを npzファイルとして保存する
        foutname = path_fspeech + "_stft"
        print("saving...", foutname)
        np.savez(os.path.join(PATH_stft, foutname + ".npz"),
                 mix=spectrogram_mix,
                 target=spectrogram_target)

def enhance_save_stft(mix_dir:str, target_dir:str, out_dir:str, is_wave:bool=True, FFT_SIZE=const.FFT_SIZE, H=const.H)->None:
    """
    音源強調用のデータセットを作成する

    Parameters
    ----------
    mix_dir(str):機械学習への入力信号(雑音付き信号)
    target_dir(str):教師データ(目的信号)
    out_dir(str):出力先
    is_wave(bool):TasNetの場合はTreu, STFTが必要な場合はFalse(UNet,LSTM)
    FFT_SIZE:FFTの窓長
    H

    Returns
    -------

    """
    """ ファイルリストの作成 """
    print("dataset")
    print(f"mix_dir:{mix_dir}")
    print(f"target_dir:{target_dir}")
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")   # 入力データ
    target_list = my_func.get_file_list(target_dir, ext=".wav") # 教師データ
    print(f"len(mix_list):{len(mix_list)}")
    print(f"len(target_list):{len(target_list)}")
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    #scaler = StandardScaler()
    #scaler = MinMaxScaler()
    for (mix_file, target_file) in tzip(mix_list, target_list):
        """ データの読み込み """
        mix_data, prm = my_func.load_wav(mix_file)  # 入力データ
        target_data, _ = my_func.load_wav(target_file)  # 教師データ
        """  データタイプの変更"""
        mix_data = mix_data.astype(np.float32)
        target_data = target_data.astype(np.float32)
        # print(f"mix_data:{mix_data.shape}")
        # print(f"target_data:{target_data.shape}")
        """ 音声長の調整 """
        min_length = min(len(mix_data), len(target_data), 128000)
        mix_data = mix_data[:min_length]
        target_data = target_data[:min_length]

        # """ 残響の影響で音声長が違う場合 """
        # mix_data = mix_data[:min([len(mix_data), len(target_data)])]
        # target_data = target_data[:min([len(mix_data), len(target_data)])]
        # mix_length = len(mix_data)
        # target_length = len(target_data)
        # print_name_type_shape("mix_length", mix_length)
        # print_name_type_shape("target_length", target_length)

        """ プリエンファシスフィルタをかける """
        # p = 0.05  # プリエンファシス係数
        # mix_data = preEmphasis(mix_data, p)
        # mix_data = mix_data.astype(np.float32)
        # target_data = preEmphasis(target_data, p)
        # target_data = target_data.astype(np.float32)
        # print("mix_data", mix_data.dtype)
        # print(mix_data.shape)
        # print("data_waveform", data_waveform.dtype)
        # print(data_waveform.shape)
        # # Scaling to -1-1
        # mix_data = mix_data / (np.iinfo(np.int16).max - 1)
        # target_data = target_data / (np.iinfo(np.int16).max - 1)
        """
        # 短時間FFTスペクトルの計算 (scipyを使用した時: WPDのPSD推定時に用いる)
        f, t, spectrogram_mix = sp.stft(mix_data, fs=16000, window="hann", nperseg=2048, noverlap=2048 - 512)
        print("spectrogram_mix", spectrogram_mix.shape)
        print(spectrogram_mix.dtype)
        print(spectrogram_mix.real ** 2 + spectrogram_mix.imag ** 2)
        f, t, spectrogram_target = sp.stft(target_data, fs=16000, window="hann", nperseg=2048, noverlap=2048 - 512)
        """

        if is_wave: # 時間領域のデータセット (TasNet,Conv-TasNet)
            # print(f"mix_data:{mix_data}")
            # print(f"target_data:{target_data}")
            """ 保存 """
            out_name, _ = my_func.get_file_name(mix_file)   # ファイル名の取得
            # print(f"saving...{out_name}")
            out_path = f"{out_dir}/{out_name}.npz"  # ファイルパスの作成
            # print(f"out:{out_path}")
            np.savez(out_path, mix=mix_data, target=target_data)    # 保存

        else:   # スペクトログラムのデータセット (UNet,LSTM)
            """ 短時間FFTスペクトルの計算 """
            spectrogram_mix = np.abs(stft(mix_data, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)).astype(np.float32)
            spectrogram_target = np.abs(stft(target_data, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)).astype(np.float32)
            """ データの整形 [特徴量(513),音声長] → [チャンネル数(1), 特徴量(513), 音声長] """
            spectrogram_mix = spectrogram_mix[np.newaxis, :, :]
            spectrogram_target = spectrogram_target[np.newaxis, :, :]
            """ スペクトルの最大値で正規化する(0~1) """
            # norm = spectrogram_mix.max()
            # spectrogram_mix /= norm
            # spectrogram_target /= norm
            spectrogram_mix =  spectrogram_mix/spectrogram_mix.max()
            spectrogram_target = spectrogram_target/spectrogram_target.max()
            print(f"mix:{spectrogram_mix}")
            print(f"target:{spectrogram_target}")
            """ 保存 """
            out_name, _ = my_func.get_file_name(mix_file)   # ファイル名の取得
            out_path = f"{out_dir}/{out_name}_stft.npz" # ファイル名の作成
            # print(f"saving...{out_name}")
            np.savez(out_path, mix=spectrogram_mix, target=spectrogram_target)  # 保存

def separate_save_stft(mix_dir:str, target_A_dir:str, target_B_dir:str, out_dir:str)->None:
    """
    話者分離用のデータセットを作成する

    Parameters
    ----------
    mix_dir(str):入力データ
    target_A_dir(str):目的信号A
    target_B_dir(str):目的信号B
    out_dir(str):出力先

    Returns
    -------
    None
    """
    """ ファイルリストの作成 """
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")
    target_A_list = my_func.get_file_list(target_A_dir, ext=".wav")
    target_B_list = my_func.get_file_list(target_B_dir, ext=".wav")
    # print(f"len(mix_list):{len(mix_list)}")
    # print(f"len(target_A_list):{len(target_A_list)}")
    # print(f"len(target_B_list):{len(target_B_list)}")

    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    #scaler = StandardScaler()
    #scaler = MinMaxScaler()
    for (mix_file, target_A_file, target_B_file) in tzip(mix_list, target_A_list, target_B_list):
        """ データの読み込み """
        """ waveを利用して読み込む """
        mix_data, _ = my_func.load_wav(mix_file)
        target_A_data, _ = my_func.load_wav(target_A_file)
        target_B_data, _ = my_func.load_wav(target_B_file)
        """ ファイルタイプの変更 """
        mix_data = mix_data.astype(np.float32)
        target_A_data = target_A_data.astype(np.float32)
        target_B_data = target_B_data.astype(np.float32)
        """ 音声長の調整 """
        mix_length = len(mix_data)
        target_A_length = len(target_A_data)
        target_B_length = len(target_B_data)
        if mix_length > 128000:
            mix_data = mix_data[:128000]
        if target_A_length > 128000:
            target_A_data = target_A_data[:128000]
        if target_B_length > 128000:
            target_B_data = target_B_data[:128000]
        # プリエンファシスフィルタをかける
        #p = 0.1  # プリエンファシス係数
        #target_A_data = preEmphasis(target_A_data, p)
        #target_A_data = target_A_data.astype(np.float32)
        """ 目的信号AとBを1つの配列にまとめる """
        target_data = np.stack([target_A_data, target_B_data])  # waveの時
        # print(f"target_data.shape:{target_data.shape}")         # [2,音声長]
        """ 保存 """
        out_name, _ = my_func.get_file_name(mix_file)  # ファイル名の取得
        # print(f"saving...{out_name}")
        # print(f"mix_data.shape:{mix_data.shape}")
        # print(f"target_data.shape:{target_data.shape}")
        out_path = f"{out_dir}/{out_name}.npz" # ファイルパスの作成
        np.savez(out_path, mix=mix_data, target=target_data)        # 保存


def separate_dataset_csv(csv_path:str, out_dir:str)->None:
    """
    話者分離用のデータセットを作成する

    Parameters
    ----------
    mix_dir(str):入力データ
    target_A_dir(str):目的信号A
    target_B_dir(str):目的信号B
    out_dir(str):出力先

    Returns
    -------
    None
    """
    """ ファイルリストの作成 """
    with open(csv_path, mode="r", newline="") as csv_file:
        data = [row for row in csv.reader(csv_file)]    # csvファイルからデータの取得
        del data[0] # ヘッダーの削除
        data = np.array(data)
    # print(f"len(mix_list):{len(mix_list)}")
    # print(f"len(target_A_list):{len(target_A_list)}")
    # print(f"len(target_B_list):{len(target_B_list)}")

    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    #scaler = StandardScaler()
    #scaler = MinMaxScaler()
    for (mix_file, target_A_file, target_B_file) in tzip(data[:, 0], data[:, 1], data[:, 2]):
        """ データの読み込み """
        """ waveを利用して読み込む """
        mix_data, _ = my_func.load_wav(mix_file)
        target_A_data, _ = my_func.load_wav(target_A_file)
        target_B_data, _ = my_func.load_wav(target_B_file)
        """ ファイルタイプの変更 """
        mix_data = mix_data.astype(np.float32)
        target_A_data = target_A_data.astype(np.float32)
        target_B_data = target_B_data.astype(np.float32)
        """ 音声長の調整 """
        # mix_length = len(mix_data)
        # target_A_length = len(target_A_data)
        # target_B_length = len(target_B_data)
        # if mix_length > 128000:
        #     mix_data = mix_data[:128000]
        # if target_A_length > 128000:
        #     target_A_data = target_A_data[:128000]
        # if target_B_length > 128000:
        #     target_B_data = target_B_data[:128000]
        min_length = min(len(mix_data), len(target_A_data), len(target_B_data), 128000)
        mix_data = mix_data[:min_length]
        target_A_data = target_A_data[:min_length]
        target_B_data = target_B_data[:min_length]
        # プリエンファシスフィルタをかける
        #p = 0.1  # プリエンファシス係数
        #target_A_data = preEmphasis(target_A_data, p)
        #target_A_data = target_A_data.astype(np.float32)
        """ 目的信号AとBを1つの配列にまとめる """
        target_data = np.stack([target_A_data, target_B_data])  # waveの時
        # print(f"target_data.shape:{target_data.shape}")         # [2,音声長]
        """ 保存 """
        out_name, _ = my_func.get_file_name(mix_file)  # ファイル名の取得
        print(f"saving...{out_name}")
        # print(f"mix_data.shape:{mix_data.shape}")
        # print(f"target_data.shape:{target_data.shape}")
        out_path = f"{out_dir}/{out_name}.npz" # ファイルパスの作成
        np.savez(out_path, mix=mix_data, target=target_data)        # 保存

def split_data(input_data:list, channel:int=0)->list:
    """
    引数で受け取ったtensor型の配列の形状を変形させる[1,音声長×チャンネル数]->[チャンネル数,音声長]

    Parameters
    ----------
    input_data(list[int]):分割する前のデータ[1, 音声長*チャンネル数]
    channels(int):チャンネル数(分割する数)

    Returns
    -------
    split_data(list[float]):分割した後のデータ[チャンネル数, 音声長]
    """
    # print("\nsplit_data")    # 確認用
    """ エラー処理 """
    if channel <= 0:   # channelsの数が0の場合or指定していない場合
        raise ValueError("channels must be greater than 0.")

    # print(f"type(in_tensor):{type(in_tensor)}") # 確認用 # torch.Tensor
    # print(f"in_tensor.shape:{in_tensor.shape}") # 確認用 # [1,音声長×チャンネル数]

    """ 配列の要素数を取得 """
    n = input_data.shape[-1]  # 要素数の取得
    # print(f"n:{n}")         # 確認用 # n=音声長×チャンネル数
    if n % channel != 0:   # 配列の要素数をchannelsで割り切れないとき = チャンネル数が間違っているとき
        raise ValueError("Input array size must be divisible by the number of channels.")

    """ 配列の分割 """
    length = n // channel   # 分割後の1列当たりの要素数を求める
    # print(f"length:{length}")   # 確認用 # 音声長
    trun_input = input_data.T   # 転置
    # print_name_type_shape("trun_tensor", trun_tensor)
    split_input = trun_input.reshape(-1, length)    # 分割
    # print_name_type_shape("split_tensor", split_input) # 確認用 # [チャンネル数, 音声長]
    # print("split_data\n")    # 確認用
    return split_input

def addition_data(input_data:ndarray, channel:int=0, delay:int=1)-> ndarray[Any, dtype[floating[_64Bit]]]:
    """ 1chの信号を遅延・減衰 (減衰率はテキトー) させる

    Parameters
    ----------
    input_data:  1chの音声データ
    channel: 拡張したいch数
    delay: どれぐらい遅延させるか

    Returns
    -------

    """
    """ エラー処理 """
    if channel <= 0:  # channelsの数が0の場合or指定していない場合
        raise ValueError("channels must be greater than 0.")
    result = np.zeros((channel, len(input_data)))
    # print("result:", result.shape)
    # print(result)
    """ 遅延させるサンプル数を指定 """
    sampling_rate = 16000
    win = 2
    window_size = sampling_rate * win // 1000  # ConvTasNetの窓長と同じ
    delay_sample = window_size  # ConvTasNetの窓長と同じ
    # delay_sample = 1    # 1サンプルだけずらす

    """ 1ch目を基準に遅延させる """
    for i in range(channel):
        result[i, delay_sample*i:] = input_data[:len(input_data)-delay_sample*i]  # 1サンプルづつずらす 例は下のコメントアウトに記載
        result[i,:] = result[i,:] * (1/2**i)   # 音を減衰させる
        if i >= 2:
            result[i,:] = -1 * result[i,:]
    """
    例
    入力：[1,2,3,4,5]
    出力：
    [[1,2,3,4,5],
     [0,1,2,3,4],
     [0,0,1,2,3],
     [0,0,0,3,4],]
    """
    """ 線形アレイを模倣した遅延 """
    # result[0, delay_sample:] = input_data[:len(input_data) - delay_sample]
    # result[1, :] = input_data
    # result[2, :] = input_data
    # result[-1, delay_sample:] = input_data[:len(input_data) - delay_sample]

    return result


""" 音の減衰を考慮して行うデータ拡張 """

def multi_channel_dataset(mix_dir:str, target_dir:str, out_dir:str, channel:int)->None:
    """
    多チャンネルのデータから1chのデータセットを作成する(教師データは1ch)

    Parameters
    ----------
    mix_dir(str):入力データのパス
    target_dir(str):目的信号のパス
    out_dir(str):出力先のパス
    num_mic(int):チャンネル数

    Returns
    -------
    None
    """
    print(f"out_dir:{out_dir}")
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    """ ファイルリストの作成 """
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")
    target_list = my_func.get_file_list(target_dir, ext=".wav")
    # print(f"len(mix_list):{len(mix_list)}")       # 確認用
    # print(f"len(target_list):{len(target_list)}") # 確認用
    for mix_file, target_file in tzip(mix_list, target_list):
        """ データの読み込み """
        mix_data, _ = my_func.load_wav(mix_file)          # waveでロード
        target_data, _ = my_func.load_wav(target_file)    # waveでロード
        # mix_data=np.asarray(mix_data)
        # print(f"mix_data.shape{mix_data.shape}")        # 確認用 # [1,音声長×チャンネル数]
        # print(f"mix_data:{mix_data}")                   # 確認用
        # print(f"target_data.shape{target_data.shape}")  # 確認用 # [1,音声長]
        # print(f"target_data:{target_data}")             # 確認用
        """ データの形状を変更 """
        mix_data = split_data(mix_data, channel)  # [1,音声長×チャンネル数]→[チャンネル数,音声長]
        target_data = split_data(target_data, channel)  # [1,音声長×チャンネル数]→[チャンネル数,音声長]
        # print(f"mix_data.shape{mix_data.shape}")    # 確認用 # [チャンネル数,音声長]
        # print(f"target_data.shape{target_data.shape}")    # 確認用 # [チャンネル数,音声長]
        # mix_data = mix_data.astype(np.float32)
        # data_waveform = mix_data[np.newaxis, :]
        # data_waveform = torch.from_numpy(data_waveform)
        # print("data", data_waveform.dtype)
        # print("mix_data", mix_data.shape)
        """ 音声長の調整 """
        target_length = target_data.shape[1]        # 音声長の取得
        # print(f"mix_length:{mix_length}")           # 確認用
        # print(f"target_length:{target_length}")     # 確認用
        if target_length > 128000:                  # 音声長が128000以上の場合
            target_length = 128000
            target_data = target_data[:, :target_length]    # 128000までにする
        mix_data = mix_data[:, :target_length]    # targetと同じ音声長にカットする
        # print(f"mix_data:{mix_data}")
        # print(f"target_data:{target_data}")
        # print(f"mix_data.shape:{mix_data.shape}")       # 確認用 # [チャンネル数,音声長]    音声長の最大値は128000
        # print(f"target_data.shape:{target_data.shape}") # 確認用 # [1,音声長]    音声長の最大値は128000
        """ 保存 """
        out_name, _ = os.path.splitext(os.path.basename(target_file))   # ファイル名の取得
        # print(f"saving...{out_name}")
        # out_path = out_dir + out_name + ".npz"                              # ファイルパスの作成
        out_path = f"{out_dir}/{out_name}.npz"                              # ファイルパスの作成
        np.savez(out_path, mix=mix_data, target=target_data)             # 保存

def multi_channel_dataset2(mix_dir:str, target_dir:str, out_dir:str, channel:int)->None:
    """
    多チャンネルのデータから多チャンネルのデータセットを作成する(教師データも多ch)

    Parameters
    ----------
    mix_dir(str):入力データ
    target_dir(str):正解データ
    out_dir(str):出力先
    num_mic(int):チャンネル数

    Returns
    -------
    None
    """
    # print("multi channels dataset2")
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    print(f"mix_dir:{mix_dir}")
    print(f"target_dir:{target_dir}")
    """ ファイルの存在を確認 """

    """ ファイルリストの作成 """
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")
    target_list = my_func.get_file_list(target_dir, ext=".wav")
    # print(f"len(mix_list):{len(mix_list)}")       # 確認用
    # print(f"len(target_list):{len(target_list)}") # 確認用

    with tqdm(total=len(mix_list), leave=False) as prog_bar:
        for mix_file, target_file in zip(mix_list, target_list):
            # prog_bar.write(f"mix:{mix_file}")
            # prog_bar.write(f"target:{target_file}")
            """ データの読み込み """
            mix_data, _ = my_func.load_wav(mix_file)          # waveでロード
            target_data, _ = my_func.load_wav(target_file)    # waveでロード
            # mix_data=np.asarray(mix_data)
            # print(f"mix_data.shape{mix_data.shape}")        # 確認用 # [1,音声長×チャンネル数]
            # print(f"mix_data:{mix_data.dtype}")                   # 確認用
            # print(f"target_data.shape{target_data.shape}")  # 確認用 # [1,音声長]
            # print(f"target_data:{target_data.dtype}")             # 確認用
            """ データの形状を変更 """
            mix_data = split_data(mix_data, channel)  # 配列の形状を変更
            target_data = split_data(target_data, channel)  # 配列の形状を変更
            # print(f"mix_data.shape{mix_data.shape}")    # 確認用 # [チャンネル数,音声長]
            # mix_data = mix_data.astype(np.float32)
            # data_waveform = mix_data[np.newaxis, :]
            # data_waveform = torch.from_numpy(data_waveform)
            # print("data", data_waveform.dtype)
            # print("mix_data", mix_data.shape)

            """ 音声長の確認と修正 """
            min_length = min(mix_data.shape[1], target_data.shape[1], 128000)
            mix_data = mix_data[:, :min_length]  # 音声長の取得
            target_data = target_data[:, :min_length]  # 音声長の取得
            # print(f"mix_length:{mix_length}")           # 確認用
            # print(f"target_length:{target_length}")     # 確認用
            # if mix_length > 128000: # 音声長が128000以上の場合
            #     mix_data = mix_data[:, :128000] # 128000までにする
            # if target_length > 128000:  # 音声長が128000以上の場合
            #     target_data = target_data[:, :128000]   # 128000までにする
            # print(f"mix_data:{mix_data}")
            # print(f"target_data:{target_data}")
            # print(f"mix_data.shape:{mix_data.shape}")       # 確認用 # [チャンネル数,音声長]    音声長の最大値は128000
            # print(f"target_data.shape:{target_data.shape}") # 確認用 # [1,音声長]    音声長の最大値は128000
            """ 音声波形をペアで保存する """
            out_name, _ = my_func.get_file_name(target_file)    # ファイル名の取得
            # print(f"saving...{out_name}")
            # out_path = out_dir+out_name+".npz"  # ファイルパスの作成
            out_path = f"{out_dir}/{out_name}.npz"   # ファイルパスの作成
            np.savez(out_path, mix=mix_data, target=target_data)    # 保存
            prog_bar.update(1)

def multi_to_single_dataset(mix_dir:str, target_dir:str, out_dir:str, channel:int)->None:
    """
    1chの入力データを4chに拡張して(開始タイミングを遅らせることでマイク間の遅延を表現)

    :param mix_dir: 入力データのディレクトリ
    :param target_dir: 教師データのディレクトリ
    :param out_dir: データセットの出力先
    :param channel: 拡張するチャンネル数(マイク数)
    :return:
    """
    # print("multi channels dataset2")
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    print(f"mix_dir:{mix_dir}")
    print(f"target_dir:{target_dir}")
    """ ファイルの存在を確認 """

    """ ファイルリストの作成 """
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")
    target_list = my_func.get_file_list(target_dir, ext=".wav")
    # print(f"len(mix_list):{len(mix_list)}")       # 確認用
    # print(f"len(target_list):{len(target_list)}") # 確認用

    with tqdm(total=len(mix_list), leave=False) as prog_bar:
        for mix_file, target_file in zip(mix_list, target_list):
            # prog_bar.write(f"mix:{mix_file}")
            # prog_bar.write(f"target:{target_file}")
            """ データの読み込み """
            mix_data, _ = my_func.load_wav(mix_file)  # waveでロード
            target_data, _ = my_func.load_wav(target_file)  # waveでロード
            # mix_data=np.asarray(mix_data)
            # print(f"mix_data.shape{mix_data.shape}")        # 確認用 # [1,音声長×チャンネル数]
            # print(f"mix_data:{mix_data.dtype}")                   # 確認用
            # print(f"target_data.shape{target_data.shape}")  # 確認用 # [1,音声長]
            # print(f"target_data:{target_data.dtype}")             # 確認用
            # print("mix_data", mix_data.shape)

            """ 音声長の確認と修正 """
            min_length = min(mix_data.shape[0], target_data.shape[0], 128000)
            mix_data = mix_data[:min_length]  # 音声長の取得
            target_data = target_data[:min_length]  # 音声長の取得
            # print(f"mix_length:{mix_length}")           # 確認用
            # print(f"target_length:{target_length}")     # 確認用
            # if mix_length > 128000: # 音声長が128000以上の場合
            #     mix_data = mix_data[:, :128000] # 128000までにする
            # if target_length > 128000:  # 音声長が128000以上の場合
            #     target_data = target_data[:, :128000]   # 128000までにする
            # print(f"mix_data:{mix_data}")
            # print(f"target_data:{target_data}")
            # print(f"mix_data.shape:{mix_data.shape}")       # 確認用 # [チャンネル数,音声長]    音声長の最大値は128000
            # print(f"target_data.shape:{target_data.shape}") # 確認用 # [1,音声長]    音声長の最大値は128000
            """ データの形状を変更 """
            mix_data = addition_data(mix_data, channel)  # 配列の形状を変更
            # target_data = np.vstack((target_data, target_data, target_data, target_data))  # 配列の形状を変更
            target_data = addition_data(target_data, channel)  # 配列の形状を変更
            # print(f"mix_data.shape{mix_data.shape}")    # 確認用 # [チャンネル数,音声長]
            # mix_data = mix_data.astype(np.float32)
            # data_waveform = mix_data[np.newaxis, :]
            # data_waveform = torch.from_numpy(data_waveform)
            # print("data", data_waveform.dtype)
            """ 音声波形をペアで保存する """
            out_name, _ = my_func.get_file_name(target_file)  # ファイル名の取得
            # print(f"saving...{out_name}")
            # out_path = out_dir+out_name+".npz"  # ファイルパスの作成
            out_path = f"{out_dir}/{out_name}.npz"  # ファイルパスの作成
            np.savez(out_path, mix=mix_data, target=target_data)  # 保存
            prog_bar.update(1)

def multi_to_single_wavfile(mix_dir:str, out_dir:str, channel:int)->None:
    """
    1chの入力データを4chに拡張して(開始タイミングを遅らせることでマイク間の遅延を表現)

    :param mix_dir: 入力データのディレクトリ
    :param out_dir: データセットの出力先
    :param channel: 拡張するチャンネル数(マイク数)
    :return:
    """
    # print("multi channels dataset2")
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    print(f"mix_dir:{mix_dir}")
    """ ファイルの存在を確認 """

    """ ファイルリストの作成 """
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")
    # print(f"len(mix_list):{len(mix_list)}")       # 確認用
    # print(f"len(target_list):{len(target_list)}") # 確認用

    with tqdm(total=len(mix_list), leave=False) as prog_bar:
        for mix_file in mix_list:
            # prog_bar.write(f"mix:{mix_file}")
            # prog_bar.write(f"target:{target_file}")
            """ データの読み込み """
            mix_data, prm = my_func.load_wav(mix_file)  # waveでロード
            # mix_data=np.asarray(mix_data)
            # print(f"mix_data.shape{mix_data.shape}")        # 確認用 # [1,音声長×チャンネル数]
            # print(f"mix_data:{mix_data.dtype}")                   # 確認用
            # print(f"target_data.shape{target_data.shape}")  # 確認用 # [1,音声長]
            # print(f"target_data:{target_data.dtype}")             # 確認用
            # print("mix_data", mix_data.shape)

            """ 音声長の確認と修正 """
            min_length = min(mix_data.shape[0], 128000)
            mix_data = mix_data[:min_length]  # 音声長の取得
            # print(f"mix_length:{mix_length}")           # 確認用
            # print(f"target_length:{target_length}")     # 確認用
            # if mix_length > 128000: # 音声長が128000以上の場合
            #     mix_data = mix_data[:, :128000] # 128000までにする
            # if target_length > 128000:  # 音声長が128000以上の場合
            #     target_data = target_data[:, :128000]   # 128000までにする
            # print(f"mix_data:{mix_data}")
            # print(f"target_data:{target_data}")
            # print(f"mix_data.shape:{mix_data.shape}")       # 確認用 # [チャンネル数,音声長]    音声長の最大値は128000
            # print(f"target_data.shape:{target_data.shape}") # 確認用 # [1,音声長]    音声長の最大値は128000
            """ データの形状を変更 """
            mix_data = addition_data(mix_data, channel)  # 配列の形状を変更
            # target_data = np.vstack((target_data, target_data, target_data, target_data))  # 配列の形状を変更
            # print(f"mix_data.shape{mix_data.shape}")    # 確認用 # [チャンネル数,音声長]
            # mix_data = mix_data.astype(np.float32)
            # data_waveform = mix_data[np.newaxis, :]
            # data_waveform = torch.from_numpy(data_waveform)
            # print("data", data_waveform.dtype)
            """ 音声波形をペアで保存する """
            out_name, _ = my_func.get_file_name(mix_file)  # ファイル名の取得
            # print(f"saving...{out_name}")
            # out_path = out_dir+out_name+".npz"  # ファイルパスの作成
            out_path = f"{out_dir}/{out_name}.wav"  # ファイルパスの作成
            my_func.save_wav(out_path=out_path, wav_data=mix_data, prm=prm)
            prog_bar.update(1)
# def multi_channle_dataset2(mix_dir, target_dir, out_dir, num_mic):
#     print("multi channels dataset2")
#     my_func.make_dir(out_dir)
#     mix_list = my_func.get_file_list(mix_dir, ext=".wav")
#     target_list = my_func.get_file_list(target_dir, ext=".wav")
#     print(f"len(mix_list):{len(mix_list)}")
#     print(f"len(target_list):{len(target_list)}")
#
#     with tqdm(total=len(mix_list), leave=False) as prog_bar:
#         for mix_file, target_file in zip(mix_list, target_list):
#             prog_bar.write(f"mix:{mix_file}")
#             prog_bar.write(f"target:{target_file}")
#             mix_data, _ = my_func.load_wav(mix_file)
#             target_data, _ = my_func.load_wav(target_file)
#             mix_data = split_data(mix_data, num_mic)
#             target_data = split_data(target_data, num_mic)
#             min_length = min(mix_data.shape[1], target_data.shape[1], 128000)
#             mix_data = mix_data[:, :min_length]
#             target_data = target_data[:, :min_length]
#             out_name, _ = my_func.get_file_name(target_file)
#             out_path = f"{out_dir}/{out_name}.npz"
#             np.savez(out_path, mix=mix_data, target=target_data)
#             prog_bar.update(1)
def multi_channel_dataset_2stage(mix_dir:str, reverbe_dir:str, target_dir:str, out_dir:str, channel:int)->None:
    """
    多チャンネルのデータから多チャンネルのデータセットを作成する(教師データも多ch)

    Parameters
    ----------
    mix_dir(str):入力データ 目的信号+残響+雑音
    target_A_dir(str):正解データ 目的信号+残響
    target_B_dir(str):正解データ 目的信号
    out_dir(str):出力先
    num_mic(int):チャンネル数

    Returns
    -------
    None
    """
    # print("multi channels dataset2")
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    print(f"mix_dir:{mix_dir}")
    print(f"reverbe_dir:{reverbe_dir}")
    print(f"target_dir:{target_dir}")
    """ ファイルの存在を確認 """

    """ ファイルリストの作成 """
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")
    reverbe_list = my_func.get_file_list(reverbe_dir, ext=".wav")
    target_dir = my_func.get_file_list(target_dir, ext=".wav")
    # print(f"len(mix_list):{len(mix_list)}")       # 確認用
    # print(f"len(reverbe_list):{len(reverbe_list)}") # 確認用

    with tqdm(total=len(mix_list), leave=False) as prog_bar:
        for mix_file, reverbe_file, target_file in zip(mix_list, reverbe_list, target_dir):
            # prog_bar.write(f"mix:{mix_file}")
            # prog_bar.write(f"target:{reverbe_file}")
            """ データの読み込み """
            mix_data, _ = my_func.load_wav(mix_file)          # waveでロード
            reverbe_data, _ = my_func.load_wav(reverbe_file)    # waveでロード
            target_data, _ = my_func.load_wav(target_file)    # waveでロード
            # mix_data=np.asarray(mix_data)
            # print(f"mix_data.shape{mix_data.shape}")        # 確認用 # [1,音声長×チャンネル数]
            # print(f"mix_data:{mix_data.dtype}")                   # 確認用
            # print(f"reverbe_data.shape{reverbe_data.shape}")  # 確認用 # [1,音声長]
            # print(f"reverbe_data:{reverbe_data.dtype}")             # 確認用
            """ データの形状を変更 """
            mix_data = split_data(mix_data, channel)  # 配列の形状を変更
            reverbe_data = split_data(reverbe_data, channel)  # 配列の形状を変更
            target_data = split_data(target_data, channel)  # 配列の形状を変更
            # print(f"mix_data.shape{mix_data.shape}")    # 確認用 # [チャンネル数,音声長]
            # mix_data = mix_data.astype(np.float32)
            # data_waveform = mix_data[np.newaxis, :]
            # data_waveform = torch.from_numpy(data_waveform)
            # print("data", data_waveform.dtype)
            # print("mix_data", mix_data.shape)

            """ 音声長の確認と修正 """
            min_length = min(mix_data.shape[1], reverbe_data.shape[1], target_data.shape[1], 128000)
            mix_data = mix_data[:, :min_length]  # 音声長の取得
            reverbe_data = reverbe_data[:, :min_length]  # 音声長の取得
            target_data = target_data[:, :min_length]  # 音声長の取得
            # print(f"mix_length:{mix_length}")           # 確認用
            # print(f"target_length:{target_length}")     # 確認用
            # if mix_length > 128000: # 音声長が128000以上の場合
            #     mix_data = mix_data[:, :128000] # 128000までにする
            # if target_length > 128000:  # 音声長が128000以上の場合
            #     reverbe_data = reverbe_data[:, :128000]   # 128000までにする
            # print(f"mix_data:{mix_data}")
            # print(f"reverbe_data:{reverbe_data}")
            # print(f"mix_data.shape:{mix_data.shape}")       # 確認用 # [チャンネル数,音声長]    音声長の最大値は128000
            # print(f"reverbe_data.shape:{reverbe_data.shape}") # 確認用 # [1,音声長]    音声長の最大値は128000
            """ 音声波形をペアで保存する """
            out_name, _ = my_func.get_file_name(target_file)    # ファイル名の取得
            # print(f"saving...{out_name}")
            # out_path = out_dir+out_name+".npz"  # ファイルパスの作成
            out_path = f"{out_dir}/{out_name}.npz"   # ファイルパスの作成
            np.savez(out_path, mix=mix_data, target=[reverbe_data, target_data])    # 保存
            prog_bar.update(1)

def process_dataset_thread(angle, ch, wav_type):
    # C:/Users/kataoka-lab/Desktop/sound_data/mix_data/sebset_DEMAND_hoth_1010dB_05sec_4ch_3cm/Back/train/noise_reverbe
    # angle = "Front"
    # subset_DEMAND_hoth_1010dB_05sec_4ch_circular_10cm
    dir_name = f"subset_DEMAND_hoth_1010dB_05sec_{ch}ch_3cm_all_angle"
    mix_dir = f"{const.MIX_DATA_DIR}/{dir_name}/train/{wav_type}"
    target_dir = f"{const.MIX_DATA_DIR}/{dir_name}/train/clean"
    out_dir = f"{const.DATASET_DIR}/{dir_name}/{wav_type}"
    # print("out_dir:", out_dir)
    # print("ch:", ch)
    # multi_channle_dataset2(mix_dir, target_dir, out_dir, ch)
    multi_to_single_dataset(mix_dir, target_dir, out_dir, ch)

def make_dataset_csv(mix_dir:str, target_dir:str, csv_path:str):
    """ データセットのパスをcsv形式で保存 """

    """ 出力ファイルの作成 """
    my_func.make_dir(csv_path)
    with open(csv_path, "w") as csv_file:  # ファイルオープン
        csv_file.write(f"mix_path, target_path\n")

    """ ファイルリストの作成 """
    print(f"mix_dir:{mix_dir}")
    print(f"target_dir:{target_dir}")
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")
    target_list = my_func.get_file_list(target_dir, ext=".wav")
    print(f"len(mix_list):{len(mix_list)}")       # 確認用
    print(f"len(target_list):{len(target_list)}") # 確認用

    with tqdm(total=len(mix_list), leave=False) as prog_bar:
        for mix_path, target_path in zip(mix_list, target_list):
            # csvファイルに書き込み
            with open(csv_path, "a") as csv_file:  # ファイルオープン
                csv_file.write(f"{mix_path},{target_path}\n")
            prog_bar.update(1)


def create_sparse_graph(k_neighbors, num_nodes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 各ノードに対してk個の近傍ノードを選択
    edge_index = torch.zeros((2, num_nodes * k_neighbors), dtype=torch.long, device=device)
    for i in range(num_nodes):
        # ランダムにk個の近傍を選択（自分自身は除外）
        neighbors = torch.randperm(num_nodes - 1, device=device)[:k_neighbors]
        neighbors[neighbors >= i] += 1  # 自分自身をスキップ
        edge_index[0, i * k_neighbors:(i + 1) * k_neighbors] = i
        edge_index[1, i * k_neighbors:(i + 1) * k_neighbors] = neighbors
    return edge_index

def GNN_dataset(mix_dir:str, target_dir:str, csv_path, out_dir:str, is_wave:bool=True, FFT_SIZE=const.FFT_SIZE, H=const.H)->None:
    """
    音源強調用のデータセットを作成する

    Parameters
    ----------
    mix_dir(str):機械学習への入力信号(雑音付き信号)
    target_dir(str):教師データ(目的信号)
    out_dir(str):出力先
    is_wave(bool):TasNetの場合はTreu, STFTが必要な場合はFalse(UNet,LSTM)
    FFT_SIZE:FFTの窓長
    H

    Returns
    -------

    """
    """ ファイルリストの作成 """
    print("dataset")
    print(f"mix_dir:{mix_dir}")
    print(f"target_dir:{target_dir}")
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")   # 入力データ
    target_list = my_func.get_file_list(target_dir, ext=".wav") # 教師データ
    print(f"len(mix_list):{len(mix_list)}")
    print(f"len(target_list):{len(target_list)}")
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    #scaler = StandardScaler()
    #scaler = MinMaxScaler()
    df = pd.read_csv(csv_path)
    for ((_, row), mix_file, target_file) in tzip(df.iterrows(),mix_list, target_list, total=len(df)):
        """ データの読み込み """
        mix_data, prm = my_func.load_wav(mix_file)  # 入力データ
        target_data, _ = my_func.load_wav(target_file)  # 教師データ
        """  データタイプの変更"""
        mix_data = mix_data.astype(np.float32)
        target_data = target_data.astype(np.float32)
        # print(f"mix_data:{mix_data.shape}")
        # print(f"target_data:{target_data.shape}")
        """ 音声長の調整 """
        min_length = min(len(mix_data), len(target_data), 128000)
        mix_data = mix_data[:min_length]
        target_data = target_data[:min_length]

        data_length = int(row["data_length"])
        edge_index = create_sparse_graph(k_neighbors=8, num_nodes=data_length)
        # print(type(edge_index))

        # """ 残響の影響で音声長が違う場合 """
        # mix_data = mix_data[:min([len(mix_data), len(target_data)])]
        # target_data = target_data[:min([len(mix_data), len(target_data)])]
        # mix_length = len(mix_data)
        # target_length = len(target_data)
        # print_name_type_shape("mix_length", mix_length)
        # print_name_type_shape("target_length", target_length)

        """ プリエンファシスフィルタをかける """
        # p = 0.05  # プリエンファシス係数
        # mix_data = preEmphasis(mix_data, p)
        # mix_data = mix_data.astype(np.float32)
        # target_data = preEmphasis(target_data, p)
        # target_data = target_data.astype(np.float32)
        # print("mix_data", mix_data.dtype)
        # print(mix_data.shape)
        # print("data_waveform", data_waveform.dtype)
        # print(data_waveform.shape)
        # # Scaling to -1-1
        # mix_data = mix_data / (np.iinfo(np.int16).max - 1)
        # target_data = target_data / (np.iinfo(np.int16).max - 1)
        """
        # 短時間FFTスペクトルの計算 (scipyを使用した時: WPDのPSD推定時に用いる)
        f, t, spectrogram_mix = sp.stft(mix_data, fs=16000, window="hann", nperseg=2048, noverlap=2048 - 512)
        print("spectrogram_mix", spectrogram_mix.shape)
        print(spectrogram_mix.dtype)
        print(spectrogram_mix.real ** 2 + spectrogram_mix.imag ** 2)
        f, t, spectrogram_target = sp.stft(target_data, fs=16000, window="hann", nperseg=2048, noverlap=2048 - 512)
        """

        if is_wave: # 時間領域のデータセット (TasNet,Conv-TasNet)
            # print(f"mix_data:{mix_data}")
            # print(f"target_data:{target_data}")
            """ 保存 """
            out_name, _ = my_func.get_file_name(mix_file)   # ファイル名の取得
            # print(f"saving...{out_name}")
            out_path = f"{out_dir}/{out_name}.npz"  # ファイルパスの作成
            # print(f"out:{out_path}")
            np.savez(out_path, mix=mix_data, target=target_data, edge_index=edge_index.to('cpu').detach().numpy())    # 保存

        else:   # スペクトログラムのデータセット (UNet,LSTM)
            """ 短時間FFTスペクトルの計算 """
            spectrogram_mix = np.abs(stft(mix_data, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)).astype(np.float32)
            spectrogram_target = np.abs(stft(target_data, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)).astype(np.float32)
            """ データの整形 [特徴量(513),音声長] → [チャンネル数(1), 特徴量(513), 音声長] """
            spectrogram_mix = spectrogram_mix[np.newaxis, :, :]
            spectrogram_target = spectrogram_target[np.newaxis, :, :]
            """ スペクトルの最大値で正規化する(0~1) """
            # norm = spectrogram_mix.max()
            # spectrogram_mix /= norm
            # spectrogram_target /= norm
            spectrogram_mix =  spectrogram_mix/spectrogram_mix.max()
            spectrogram_target = spectrogram_target/spectrogram_target.max()
            print(f"mix:{spectrogram_mix}")
            print(f"target:{spectrogram_target}")
            """ 保存 """
            out_name, _ = my_func.get_file_name(mix_file)   # ファイル名の取得
            out_path = f"{out_dir}/{out_name}_stft.npz" # ファイル名の作成
            # print(f"saving...{out_name}")
            np.savez(out_path, mix=spectrogram_mix, target=spectrogram_target)  # 保存

def GNN_dataset2(mix_dir:str, target_dir:str, csv_path, out_dir:str, is_wave:bool=True, FFT_SIZE=const.FFT_SIZE, H=const.H)->None:
    """
    音源強調用のデータセットを作成する

    Parameters
    ----------
    mix_dir(str):機械学習への入力信号(雑音付き信号)
    target_dir(str):教師データ(目的信号)
    out_dir(str):出力先
    is_wave(bool):TasNetの場合はTreu, STFTが必要な場合はFalse(UNet,LSTM)
    FFT_SIZE:FFTの窓長
    H

    Returns
    -------

    """
    """ ファイルリストの作成 """
    print("dataset")
    print(f"mix_dir:{mix_dir}")
    print(f"target_dir:{target_dir}")
    mix_list = my_func.get_file_list(mix_dir, ext=".wav")   # 入力データ
    target_list = my_func.get_file_list(target_dir, ext=".wav") # 教師データ
    print(f"len(mix_list):{len(mix_list)}")
    print(f"len(target_list):{len(target_list)}")
    """ 出力先の作成 """
    my_func.make_dir(out_dir)
    #scaler = StandardScaler()
    #scaler = MinMaxScaler()
    df = pd.read_csv(csv_path)
    for ((_, row), mix_file, target_file) in tzip(df.iterrows(),mix_list, target_list, total=len(df)):
        """ データの読み込み """
        mix_data, prm = my_func.load_wav(mix_file)  # 入力データ
        target_data, _ = my_func.load_wav(target_file)  # 教師データ
        """  データタイプの変更"""
        mix_data = mix_data.astype(np.float32)
        target_data = target_data.astype(np.float32)
        # print(f"mix_data:{mix_data.shape}")
        # print(f"target_data:{target_data.shape}")
        """ 音声長の調整 """
        min_length = min(len(mix_data), len(target_data), 128000)
        mix_data = mix_data[:min_length]
        target_data = target_data[:min_length]

        data_length = int(row["data_length"])
        edge_index = create_sparse_graph(k_neighbors=8, num_nodes=data_length)
        # print(type(edge_index))

        # """ 残響の影響で音声長が違う場合 """
        # mix_data = mix_data[:min([len(mix_data), len(target_data)])]
        # target_data = target_data[:min([len(mix_data), len(target_data)])]
        # mix_length = len(mix_data)
        # target_length = len(target_data)
        # print_name_type_shape("mix_length", mix_length)
        # print_name_type_shape("target_length", target_length)

        """ プリエンファシスフィルタをかける """
        # p = 0.05  # プリエンファシス係数
        # mix_data = preEmphasis(mix_data, p)
        # mix_data = mix_data.astype(np.float32)
        # target_data = preEmphasis(target_data, p)
        # target_data = target_data.astype(np.float32)
        # print("mix_data", mix_data.dtype)
        # print(mix_data.shape)
        # print("data_waveform", data_waveform.dtype)
        # print(data_waveform.shape)
        # # Scaling to -1-1
        # mix_data = mix_data / (np.iinfo(np.int16).max - 1)
        # target_data = target_data / (np.iinfo(np.int16).max - 1)
        """
        # 短時間FFTスペクトルの計算 (scipyを使用した時: WPDのPSD推定時に用いる)
        f, t, spectrogram_mix = sp.stft(mix_data, fs=16000, window="hann", nperseg=2048, noverlap=2048 - 512)
        print("spectrogram_mix", spectrogram_mix.shape)
        print(spectrogram_mix.dtype)
        print(spectrogram_mix.real ** 2 + spectrogram_mix.imag ** 2)
        f, t, spectrogram_target = sp.stft(target_data, fs=16000, window="hann", nperseg=2048, noverlap=2048 - 512)
        """

        if is_wave: # 時間領域のデータセット (TasNet,Conv-TasNet)
            # print(f"mix_data:{mix_data}")
            # print(f"target_data:{target_data}")
            """ 保存 """
            out_name, _ = my_func.get_file_name(mix_file)   # ファイル名の取得
            # print(f"saving...{out_name}")
            out_path = f"{out_dir}/{out_name}.npz"  # ファイルパスの作成
            # print(f"out:{out_path}")
            np.savez(out_path, mix=mix_data, target=target_data, edge_index=edge_index.to('cpu').detach().numpy())    # 保存

        else:   # スペクトログラムのデータセット (UNet,LSTM)
            """ 短時間FFTスペクトルの計算 """
            spectrogram_mix = np.abs(stft(mix_data, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)).astype(np.float32)
            spectrogram_target = np.abs(stft(target_data, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)).astype(np.float32)
            """ データの整形 [特徴量(513),音声長] → [チャンネル数(1), 特徴量(513), 音声長] """
            spectrogram_mix = spectrogram_mix[np.newaxis, :, :]
            spectrogram_target = spectrogram_target[np.newaxis, :, :]
            """ スペクトルの最大値で正規化する(0~1) """
            # norm = spectrogram_mix.max()
            # spectrogram_mix /= norm
            # spectrogram_target /= norm
            spectrogram_mix =  spectrogram_mix/spectrogram_mix.max()
            spectrogram_target = spectrogram_target/spectrogram_target.max()
            print(f"mix:{spectrogram_mix}")
            print(f"target:{spectrogram_target}")
            """ 保存 """
            out_name, _ = my_func.get_file_name(mix_file)   # ファイル名の取得
            out_path = f"{out_dir}/{out_name}_stft.npz" # ファイル名の作成
            # print(f"saving...{out_name}")
            np.savez(out_path, mix=spectrogram_mix, target=spectrogram_target)  # 保存


if __name__ == "__main__":
    print("start")


    """ 音声強調用のデータセット """
    mix_dir = f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_0505dB/train"
    out_dir = f"{const.DATASET_DIR}/subset_DEMAND_hoth_0505dB/"
    sub_dir_list = my_func.get_subdir_list(mix_dir)
    sub_dir = "noise_only"
    # print(sub_dir_list)
    # for sub_dir in sub_dir_list:
    enhance_save_stft(mix_dir=os.path.join(mix_dir, sub_dir),
                        target_dir=os.path.join(mix_dir, "clean"),
                        out_dir=os.path.join(out_dir, sub_dir),
                        is_wave=True)  # False=スペクトログラム, True=時間領域
    # GNN_dataset(mix_dir=os.path.join(mix_dir, "noise_reverbe"),
    #                 target_dir=os.path.join(mix_dir, "clean"),
    #                 csv_path="C:/Users/kataoka-lab/Desktop/sound_data/dataset/DEMAND_1ch/condition_4/condition4_data_length.csv",
    #                 out_dir=out_dir,
    #                 is_wave=True)  # False=スペクトログラム, True=時間領域

    """ 音源分離用のデータセット """
    # separate_dataset_csv(csv_path="C:/Users/kataoka-lab/Desktop/sound_data/mix_data/separate_sebset_DEMAND/train/list.csv",
    #                      out_dir="C:/Users/kataoka-lab/Desktop/sound_data/dataset/separate_sebset_DEMAND")
    # separate_save_stft(mix_dir="C:/Users/kataoka-lab/Desktop/sound_data/mix_data/separate_sebset_DEMAND/train/mix/",
    #                    target_A_dir="C:/Users/kataoka-lab/Desktop/sound_data/mix_data/separate_sebset_DEMAND/train/speeker1/",
    #                    target_B_dir="C:/Users/kataoka-lab/Desktop/sound_data/mix_data/separate_sebset_DEMAND/train/speeker2/",
    #                    out_dir="C:/Users/kataoka-lab/Desktop/sound_data/dataset/separate_sebset_DEMAND1")
    
    """ 多チャンネル用のデータセット 出力：1ch"""
    # wav_type_list = ["noise_only", "noise_reverbe", "reverbe_only"]
    # ch = 4
    # for wav_type in wav_type_list:
    #     multi_channle_dataset2(mix_dir=f"C:/Users/kataoka-lab/Desktop/sound_data/mix_data/DEMAND_hoth_1010dB_05sec_4ch/train/{wav_type}",
    #                           target_dir=f"C:/Users/kataoka-lab/Desktop/sound_data/mix_data/DEMAND_hoth_1010dB_05sec_4ch/train/clean",
    #                           out_dir=f"C:/Users/kataoka-lab/Desktop/sound_data/dataset/DEMAND_hoth_1010dB_05sec_4ch/{wav_type}",
    #                           num_mic=ch)
    # multi_channle_dataset(mix_dir="../../sound_data/Experiment/mix_data/multi_ch/training/noise_reverberation",
    #                       target_dir="../../sound_data/sample_data/speech/JA/training",
    #                       out_dir="../../sound_data/Experiment/dataset/multich_noise_reverberation_out2/",
    #                       num_mic=4)

    # wav_type_list = ["noise_only", "noise_reverbe", "reverbe_only"]
    # ch = 4
    # # angle_name_list = ["Right", "FrontRight", "Front", "FrontLeft", "Left"]
    #
    # start = time.time()
    # # for wav_type in wav_type_list:
    # #     with ThreadPoolExecutor() as executor:
    # #         executor.map(process_dataset_thread, angle_name_list, [ch] * len(angle_name_list), [wav_type] * len(angle_name_list))
    # end = time.time()
    # print(f"time:{(end - start) / 60:.2f}")

    """ 多チャンネル用のデータセット 出力：多ch"""
    # mix_dir = "C:/Users/kataoka-lab/Desktop/sound_data/mix_data/subset_DEMAND_hoth_1010dB_05sec_4ch_3cm/"
    # # target_dir = "C:/Users/kataoka-lab/Desktop/sound_data/mix_data/1ch_to_4ch_decay_all/train/clean"
    # out_dir = "C:/Users/kataoka-lab/Desktop/sound_data/dataset/subset_DEMAND_hoth_1010dB_05sec_4ch_3cm"
    # # sub_dir_list.remove("clean")
    # # sub_dir_list.remove("noise_only")
    # for angle in ["Right", "FrontRight", "Front", "FrontLeft", "Left"]:
    #     # sub_dir_list = my_func.get_subdir_list(os.path.join(mix_dir, angle))
    #     for sub_dir in ["train"]:
    #         for wav_type in ["noise_only", "noise_reverbe", "reverbe_only"]:
    #             multi_channel_dataset2(mix_dir=os.path.join(mix_dir, angle, sub_dir, wav_type),
    #                                    target_dir=os.path.join(mix_dir, angle, sub_dir, "clean"),
    #                                    out_dir=os.path.join(out_dir, wav_type),
    #                                    channel=4)

    """ 1chで収音した音を遅延させて疑似的にマルチチャンネルで録音したことにするデータセット (教師データは4ch) """
    # wav_type_list = ["noise_only", "noise_reverbe", "reverbe_only", "clean"]
    # dir_name = "subset_DEMAND_hoth_1010dB_1ch"
    # out_dir_name = "subset_DEMAND_hoth_1010dB_1ch_to_4ch_win_array"
    # # C:\Users\kataoka-lab\Desktop\sound_data\mix_data\subset_DEMAND_hoth_1010dB_1ch\subset_DEMAND_hoth_1010dB_01sec_1ch\test
    #
    #
    # # for reverbe in range(1, 6):
    # reverbe = 5
    # # C:\Users\kataoka-lab\Desktop\sound_data\mix_data\subset_DEMAND_hoth_1010dB_1ch\subset_DEMAND_hoth_1010dB_01sec_1ch\test
    # mix_dir = f"{const.MIX_DATA_DIR}/{dir_name}/subset_DEMAND_hoth_1010dB_{reverbe:02}sec_1ch/test"
    # out_dir = f"{const.MIX_DATA_DIR}/{out_dir_name}/{reverbe:02}sec/test"
    # for wav_type in wav_type_list:
    #     multi_to_single_wavfile(mix_dir=os.path.join(mix_dir, wav_type),
    #                             out_dir=os.path.join(out_dir, wav_type),
    #                             channel=4)


    """ パスをcsv形式で保存する """
    # mix_dir = "C:/Users/kataoka-lab/Desktop/sound_data/mix_data/DEMAND_hoth_1010dB_05sec_4ch/train"
    # out_dir = "C:/Users/kataoka-lab/Desktop/sound_data/dataset/DEMAND_hoth_1010dB_05sec_4ch"
    # base_name = "DEMAND_hoth_1010dB_05sec_4ch"
    #
    # wave_type_list = my_func.get_subdir_list(mix_dir)
    # for wave_type in wave_type_list:
    #     make_dataset_csv(mix_dir=os.path.join(mix_dir, wave_type),
    #                      target_dir=os.path.join(mix_dir, "clean"),
    #                      csv_path=os.path.join(out_dir, f"{wave_type}_{base_name}.csv"))

