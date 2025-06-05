"""
construction
設定

"""
import os.path

DATASET_KEY_TARGET = 'target'
DATASET_KEY_MIXDOWN = 'mix'

DIR_KEY_TARGET = 'target'
DIR_KEY_NOISE = 'noise'
DIR_KEY_MIX = 'mix'

SAUND_DATA_DIR = 'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\'
SAMPLE_DATA_DIR = os.path.join(SAUND_DATA_DIR, 'sample_data')
MIX_DATA_DIR = os.path.join(SAUND_DATA_DIR, 'mix_data')
DATASET_DIR = os.path.join(SAUND_DATA_DIR, 'dataset')

RESULT_DIR = 'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\RESULT\\'
OUTPUT_WAV_DIR = os.path.join(RESULT_DIR, 'output_wav')
LOG_DIR = os.path.join(RESULT_DIR, 'log')
PTH_DIR = os.path.join(RESULT_DIR, 'pth')
EVALUATION_DIR = os.path.join(RESULT_DIR, 'evaluation')


SR = 16000  # サンプリング周波数
FFT_SIZE = 1024 # FFTのサイズ
H = 256 # 窓長

BATCHSIZE = 1  # バッチサイズ
PATCHLEN = 16   # パッチサイズ
EPOCH = 100 # 学習回数
