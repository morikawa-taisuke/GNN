# coding:utf-8
""" 汎用的な関数を定義 """

import os
import numpy as np
import wave
import array
import datetime
import torchaudio

# from mymodule import const
# import const.SR as SR
#from BF_ConvTasNet import BF_config as conf

SR = 16000

""" ファイル/ディレクトリ関係の関数 """
def get_file_name(path:str)->tuple:    #->list[str, str]
    """
    パスからファイル名のみを取得

    get_file_name('./dir1/dir2/file_name.ext') -> 'file_name', 'ext'
    Parameters
    ----------
    path(str):取得するファイルのパス

    Returns
    -------
    file_name(str):ファイル名
    ext(str):拡張子
    """
    file_name, ext = os.path.splitext(os.path.basename(path))
    # print(f'file_name:{type(file_name)}')
    # print(f'ext:{type(ext)}')
    return file_name, ext

def get_dir_name(path:str)->str:
    """
    パスから親ディレクトリを取得

    get_dir_name('./dir1/dir2/file_name.ext') -> './dir1/dir2/'
    Parameters
    ----------
    path(str):目的のパス

    Returns
    -------
    dir_path:親ディレクトリのパス
    """
    dir_path = os.path.dirname(path)
    # print(f'path:{path}')
    # print(f'dirname:{dirname}')
    return dir_path

def make_dir(path:str)->None:
    """
    目的のディレクトリを作成(ファイル名が含まれる場合,親ディレクトリを作成)

    Parameters
    ----------
    path(str):作成するディレクトリのパス

    Returns
    -------
    None
    """
    """ 作成するディレクトリが存在するかどうかを確認する """
    _, ext = os.path.splitext(path) # dir_pathの拡張子を取得
    if len(ext) == 0:   # ディレクトリのみ場合
        os.makedirs(path, exist_ok=True)
    elif not (ext) == 0:    # ファイル名を含む場合
        os.makedirs(get_dir_name(path), exist_ok=True)

def get_subdir_list(dir_path:str)->list:
    """
    指定したディレクトリの子ディレクトリのディレクトリ名のみをリストアップ

    dir
    |
    |----dir1
    |
    -----dir2

    get_dir_list('./dir')->['dir1', 'dir2']
    Parameters
    ----------
    path(str):

    Returns
    -------

    """
    return [file_path for file_path in os.listdir(dir_path) if os.path.splitext(file_path)[1] == '']

def get_file_list(dir_path:str, ext:str='.wav') -> list:
    """
    指定したディレクトリ内の任意の拡張子のファイルをリストアップ

    Parameters
    ----------
    dir_path(str):ディレクトリのパス
    ext(str):拡張子

    Returns
    -------
    list[str]
    """
    if os.path.isdir(dir_path):
        return [f'{dir_path}/{file_path}' for file_path in os.listdir(dir_path) if os.path.splitext(file_path)[1] == ext]
    else:
        return [dir_path]
    
def path_slice(path:str)->str:
    """
    パスの最後を取得する

    path_slice('./dir/subdir/file.ext') -> 'file.ext'
    Parameters
    ----------
    path(str):パス

    Returns
    -------
    path_slice(str):pathの最後
    """
    path_slice = path.split('\\')  # pathを'\\'でスライス
    return path_slice[-1]

def get_wave_list_from_subdir(dir_path:str)->list:
    """
    サブディレクトリに含まれる音源ファイルをすべてリストアップ

    Parameters
    ----------
    dir_path(str):探索する親ディレクトリ

    Returns
    -------
    file_list(list[str]):音源ファイルリスト
    """
    subdir_list = get_subdir_list(dir_path)  # サブディレクトリのリストアップ
    file_list = []
    for dir in subdir_list:
        list = get_file_list(dir_path=dir, ext='.wav')   # サブディレクトリの音源ファイルをリスト化
        file_list.append(list)  # file_listに追加
    return file_list


""" 音源関係の関数 """
def load_wav(wave_path:str, sample_rate:int= SR)->tuple:
    """
    音声ファイルの読み込み

    Parameters
    ----------
    wav_path(str):パス

    Returns
    -------

    """
    with wave.open(wave_path, "r") as wav:
        prm = wav.getparams()   # パラメータオブジェクト
        wave_data = wav.readframes(wav.getnframes())    # 音声データの読み込み(バイナリ形式)
        wave_data = np.frombuffer(wave_data, dtype=np.int16)    # 振幅に変換
        # wave_data = wave_data / np.iinfo(np.int16).max  # 最大値で正規化
        wave_data = wave_data.astype(np.float64)
        # if not prm.framerate == sample_rate:    # wavファイルのサンプリング周波数が任意のサンプリング周波数と違う場合
        #     prm.amplitude = resample(np.astype(np.float64), prm.framerate, sample_rate)  # サンプリング周波数をあわせる
    return wave_data, prm

def save_wav(out_path:str, wav_data, prm:object, sample_rate:int= SR)->None:
    """
    wav_dataの保存

    Parameters
    ----------
    out_path(str):出力パス
    wav_data(list[float]):音源データ
    prm(object):音源データのパラメータ
    sample_rate(int):サンプリング周波数

    Returns
    -------
    None
    """
    # wav_file = wave.Wave_write(out_path)
    # wav_file.setparams(prm)
    # wav_file.setframerate(sample_rate)
    # #wav_file.writeframes(array.array('h', wav.astype(np.int16)).tostring())
    # wav_file.writeframes(array.array('h', wav_data.astype(np.int16)).tobytes())
    # wav_file.close()

    # print(f'out_path:{out_path}')
    make_dir(path=out_path) # 保存先の作成
    with wave.open(out_path, "wb") as wave_file:    # ファイルオープン
        wave_file.setparams(prm)    # パラメータのセット
        wave_file.setframerate(sample_rate) # サンプリング周波数の上書き
        # wave_file.writeframes(array.array('h', wav_data.astype(np.int16)).tobytes())    # データの書き込み
        wave_file.writeframes(wav_data.astype(np.int16))

def load_tensor(wave_path:str):
    wave_data, _ = torchaudio.load(wave_path)
    return wave_data

def save_tensor(out_path:str, wav_data, sample_rate:int= SR):
    # print()
    torchaudio.save(out_path, wav_data, sample_rate=sample_rate)


""" 記録関係 """
def record_loss(file_name, text):
    with open(file_name, 'a') as out_file:  # ファイルオープン
        out_file.write(f'{text}\n')  # 書き込み

def get_max_row(sheet):
    """ 引数で受け取ったシートの最終行(空白行)を取得する
    
    :param sheet: 最終行を取得するシート
    :return max_row: 最終行(値がない行 -> 書式設定されていても空白の場合，最終行として認識しないようにする)
    """
    max_row = sheet.max_row + 1
    max_column = sheet.max_column + 1
    # print(f'max_row:{max_row}')
    # print(f'max_column:{max_column}')
    for row in range(max_row, 1, -1):
        for column in range(1, max_column):
            # print(f'[{row}:{column}] = {sheet.cell(row=row, column=column).value}')
            if sheet.cell(row=row, column=column).value != None:
                max_row = row
                return max_row+1
    return 1

def get_now_time():
    now = datetime.datetime.now()  # 今日の日付を取得
    now_text = f'{now.month}m{now.day}d{now.hour}h{now.minute}min'
    return now_text

if __name__ == '__main__':
    print('my_func')