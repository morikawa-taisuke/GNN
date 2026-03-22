
#順列計算に使用
import itertools
import time
import math
import wave as wave
#import pyroomacoustics as pa
import numpy as np
import scipy.signal as sp
import scipy as scipy
import os
import array
import argparse
import random
import librosa
import librosa.display
from librosa.core import resample
from librosa.util import find_files
import sys
#順列計算に使用
import itertools
import pylab
#from sympy import *
from scipy import integrate

from scipy.signal import stft,get_window,correlate,resample
from scipy.linalg import solve_toeplitz,toeplitz
import scipy
#import pesq as pypesq # https://github.com/ludlows/python-pesq

#from scipy.signal import firls,kaiser,upfirdn
from fractions import Fraction

def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

#2バイトに変換してファイルに保存
#signal: time-domain 1d array (float)
#file_name: 出力先のファイル名
#sample_rate: サンプリングレート
def write_file_from_time_signal(signal,file_name,sample_rate):
    #2バイトのデータに変換
    signal=signal.astype(np.int16)

    #waveファイルに書き込む
    wave_out = wave.open(file_name, 'w')

    #モノラル:1、ステレオ:2
    wave_out.setnchannels(1)

    #サンプルサイズ2byte
    wave_out.setsampwidth(2)

    #サンプリング周波数
    wave_out.setframerate(sample_rate)

    #データを書き込み
    wave_out.writeframes(signal)

    #ファイルを閉じる
    wave_out.close()

#SNRをはかる
#desired: 目的音、Lt
#out:　雑音除去後の信号 Lt
def calculate_snr(desired,out):
    wave_length=np.minimum(np.shape(desired)[0],np.shape(out)[0])

    #消し残った雑音
    desired=desired[:wave_length]
    out=out[:wave_length]
    noise=desired-out
    snr=10.*np.log10(np.sum(np.square(desired))/np.sum(np.square(noise)))

    return(snr)

def get_wave_filelist(path):
    if os.path.isdir(path):
        # 入力がディレクトリーの場合、ファイルリストをつくる
        filelist = find_files(path, ext="wav", case_sensitive=True)
    else:
        # 入力が単一ファイルの場合
        filelist = [path]
    print('number of file', len(filelist))

    return filelist


def load_wav(path, SR=16000):
    wav = wave.open(path, "r")
    # print("wav", wav.shape)
    prm = wav.getparams()
    # print("prm", prm.shape)
    buffer = wav.readframes(wav.getnframes())
    # print("buffer", buffer.shape)
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    # print("amptitude", amptitude.shape)
    # sr = prm.framerate
    # if not sr == SR:
    # サンプリングレートをあわせる
    # amptitude = resample(amptitude.astype(np.float64), sr, SR)

    return amptitude, prm


def save_wav(path, wav, prm, SR=16000):
    f = wave.Wave_write(path)
    f.setparams(prm)
    # f.setframerate(SR)
    f.writeframes(array.array('h', wav.astype(np.int16)).tostring())
    f.close()

def get_fname(path):
    fname, ext = os.path.splitext(os.path.basename(path))
    return fname, ext

def lpc2cep(a):
    #
    # converts prediction to cepstrum coefficients
    #
    # Author: Philipos C. Loizou

    M=len(a)
    cep=np.zeros((M-1,))

    cep[0]=-a[1]

    for k in range(2,M):
        ix=np.arange(1,k)
        vec1=cep[ix-1]*a[k-1:0:-1]*(ix)
        cep[k-1]=-(a[k]+np.sum(vec1)/k)
    return cep

def lpcoeff(speech_frame, model_order):
    eps = np.finfo(np.float64).eps
    # ----------------------------------------------------------
    # (1) Compute Autocorrelation Lags
    # ----------------------------------------------------------
    winlength = max(speech_frame.shape)
    R = np.zeros((model_order + 1,))
    for k in range(model_order + 1):
        if k == 0:
            R[k] = np.sum(speech_frame[0:] * speech_frame[0:])
        else:
            R[k] = np.sum(speech_frame[0:-k] * speech_frame[k:])

    # R=scipy.signal.correlate(speech_frame,speech_frame)
    # R=R[len(speech_frame)-1:len(speech_frame)+model_order]
    # ----------------------------------------------------------
    # (2) Levinson-Durbin
    # ----------------------------------------------------------
    a = np.ones((model_order,))
    a_past = np.ones((model_order,))
    rcoeff = np.zeros((model_order,))
    E = np.zeros((model_order + 1,))

    E[0] = R[0]

    for i in range(0, model_order):
        a_past[0:i] = a[0:i]

        sum_term = np.sum(a_past[0:i] * R[i:0:-1])

        if E[i] == 0.0:  # prevents zero division error, numba doesn't allow try/except statements
            rcoeff[i] = np.inf
        else:
            rcoeff[i] = (R[i + 1] - sum_term) / (E[i])

        a[i] = rcoeff[i]
        # if i==0:
        #    a[0:i] = a_past[0:i] - rcoeff[i]*np.array([])
        # else:
        if i > 0:
            a[0:i] = a_past[0:i] - rcoeff[i] * a_past[i - 1::-1]

        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]

    acorr = R
    refcoeff = rcoeff
    lpparams = np.ones((model_order + 1,))
    lpparams[1:] = -a
    return (lpparams, R)

def cepstrum_distance(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)

    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples

    if fs < 10000:
        P = 10  # LPC Analysis Order
    else:
        P = 16  # this could vary depending on sampling frequency.

    C = 10 * np.sqrt(2) / np.log(10)

    numFrames = int(clean_length / skiprate - (winlength / skiprate))  # number of frames

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extract_overlapped_windows(
        clean_speech[0:int(numFrames) * skiprate + int(winlength - skiprate)], winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extract_overlapped_windows(
        processed_speech[0:int(numFrames) * skiprate + int(winlength - skiprate)], winlength, winlength - skiprate,
        hannWin)
    distortion = np.zeros((numFrames,))

    for ii in range(numFrames):
        A_clean, R_clean = lpcoeff(clean_speech_framed[ii, :], P)
        A_proc, R_proc = lpcoeff(processed_speech_framed[ii, :], P)

        C_clean = lpc2cep(A_clean)
        C_processed = lpc2cep(A_proc)
        distortion[ii] = min((10, C * np.linalg.norm(C_clean - C_processed)))

    IS_dist = distortion
    alpha = 0.95
    IS_len = round(len(IS_dist) * alpha)
    IS = np.sort(IS_dist)
    cep_mean = np.mean(IS[0: IS_len])
    return cep_mean

def lesson(clean, noise, files):
    clean_files = get_wave_filelist(clean)
    noise_files = get_wave_filelist(noise)
    print('clean files length =', len(clean_files))
    print('noise files length =', len(noise_files))
    #start = random.randint(0, 960000 - 128000)

    #f = open('../../data_sample/tasnet-train/cd_'+ str(files) +'.txt', 'w')
    f = open('../../sound_data/' + str(files) + '.csv', 'w')

    for clean_file, noise_file in zip(clean_files, noise_files):
        #print("clean_file", clean_file)
        wav = wave.open(clean_file)
        fs = wav.getframerate()
        data = wav.readframes(wav.getnframes())
        data = np.frombuffer(data, dtype=np.int16)
        clean_wav = data / np.iinfo(np.int16).max

        wav = wave.open(noise_file)
        data = wav.readframes(wav.getnframes())
        data = np.frombuffer(data, dtype=np.int16)
        noise_wav = data / np.iinfo(np.int16).max

        max = len(clean_wav)
        noise_wav = noise_wav[:max]

        p_max = np.max(clean_wav)
        noise_wav = noise_wav * (p_max/np.max(noise_wav))

        a = cepstrum_distance(clean_wav, noise_wav, fs)

        #print("cepstrum distance", str(a))
        f.write(str(a))
        f.write("\n")

    f.close()

if __name__ == '__main__':
    lesson("../../sound_data/ConvTasNet/test/target",
           "../../sound_data/ConvTasNet/test/test",
           "ConvtasNet/evaluation/cepsdist_befor_spectorogram")
    lesson("../../sound_data/ConvTasNet/test/target",
           "../../sound_data/ConvTasNet/result/test",
           "ConvtasNet/evaluation/cepsdist_after_spectorogram")

    lesson("../../sound_data/UNet/test/target",
           "../../sound_data/UNet/test/test",
           "ConvtasNet/evaluation/cepsdist_befor_wave")
    lesson("../../sound_data/UNet/test/target",
           "../../sound_data/UNet/result/test",
           "ConvtasNet/evaluation/cepsdist_after_wave")


    #lesson("../../data_sample/reverb_0.7_snr20_a/clean", "../../data_sample/reverb_0.7_snr20_a/dnn-wpe", "dnn-wpe")
    #lesson("../../data_sample/reverb_0.7_snr20_a/clean", "../../data_sample/reverb_0.7_snr20_a/wpd", "wpd")
    #lesson("../../data_sample/tasnet-train/clean_no_noise_and_reverb", "../../data_sample/tasnet-train/test",
    #       "clean_no_noise_and_reverb_mix")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb1020_snr20/clean",
    #       "../../data_sample/reverb_0.3_snr20/reverb1020_snr20/mix_tasnet_sisnr_dereverb", "mix_tasnet_sisnr_dereverb")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb1020_snr20/clean",
    #       "../../data_sample/reverb_0.3_snr20/reverb1020_snr20/mix_tasnet_sisdr_dereverb", "mix_tasnet_sisdr_dereverb")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb1020_snr20/clean",
    #       "../../data_sample/reverb_0.3_snr20/reverb1020_snr20/mix_tasnet_mse_dereverb", "mix_tasnet_mse_dereverb")
    #lesson("../../data_sample/eval/clean",
    #       "../../data_sample/eval/tasnet_sisdr-sisnr_reverb", "tasnet_sisdr-sisnr_reverb")
    #lesson("../../data_sample/eval/clean",
    #       "../../data_sample/eval/tasnet_sisdr_reverb", "tasnet_sisdr_reverb")
    #lesson("../../data_sample/eval/clean",
    #       "../../data_sample/eval/tasnet_sisnr_reverb", "tasnet_sisnr_reverb")
    #lesson("../../data_sample/eval/clean",
    #       "../../data_sample/eval/dnn-wpe", "dnn-wpe")
    #lesson("../../data_sample/eval/clean",
    #       "../../data_sample/eval/test", "test")
    #lesson("../../data_sample/eval/clean",
    #       "../../data_sample/eval/tasnet_sisdr-sisnr_mvdr", "tasnet_sisdr-sisnr_mvdr")
    #lesson("../../data_sample/eval/clean",
    #       "../../data_sample/eval/tasnet_sisdr_mvdr", "tasnet_sisdr_mvdr")
    #lesson("../../data_sample/eval/clean",
    #       "../../data_sample/eval/tasnet_sisnr_mvdr", "tasnet_sisnr_mvdr")
    #lesson("../../data_sample/test/clean",
    #       "../../data_sample/test/tasnet-mse", "tasnet-mse")
    #lesson("../../data_sample/test/clean",
    #       "../../data_sample/test/tasnet-sisnr075-mse025", "tasnet-sisnr075-mse025")
    #lesson("../../data_sample/test/clean",
    #       "../../data_sample/test/tasnet_clean_02_pres005-sisdr-sisnr", "tasnet_clean_02_pres005-sisdr-sisnr")
    #lesson("../../data_sample/test/clean",
    #       "../../data_sample/test/tasnet_clean_02-sisdr-sisnr", "tasnet_clean_02-sisdr-sisnr")
    #lesson("../../data_sample/test/clean",
    #       "../../data_sample/test/tasnet_pres05-sisdr-sisnr2", "tasnet_pres05-sisdr-sisnr2")
    #lesson("../../data_sample/test/clean",
    #       "../../data_sample/test/tasnet_pres097-sisdr-sisnr2", "tasnet_pres097-sisdr-sisnr2")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb1020_snr20_direct/clean",
    #       "../../data_sample/reverb_0.3_snr20/reverb1020_snr20_direct/clean-noise", "clean-noise")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_direct/clean_no_noise_and_reverb",
    #       "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_direct/tasnet-sisnr-mvdr", "tasnet-sisnr-mvdr")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_direct/clean_no_noise_and_reverb",
    #       "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_direct/tasnet-sisdr-sisnr-mvdr", "tasnet-sisdr-sisnr-mvdr")

    #lesson("../../data_sample/reverb_0.3_snr20/reverb1020_snr20/clean",
    #      "../../data_sample/reverb_0.3_snr20/reverb1020_snr20/test", "test")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb1020_snr20/clean",
    #      "../../data_sample/reverb_0.3_snr20/reverb1020_snr20/mix_tasnet_MSE", "mix_tasnet_MSE")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/clean", "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/test",
    #       "test")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/clean", "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/wpd(dnn-wpe)",
    #       "wpd(dnn-wpe)")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/clean", "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/wpd(dnn-wpe-max)",
    #       "wpd(dnn-wpe-max)")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/clean", "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/wpd(tasnet)",
    #       "wpd(tasnet)")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/clean", "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/dnn-wpe",
    #       "dnn-wpe")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/clean", "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/dnn-wpe-max",
    #       "dnn-wpe-max")
    #lesson("../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/clean", 
    #       "../../data_sample/reverb_0.3_snr20/reverb_0.3_snr20_k/tasnet",
    #       "tasnet")