import csv
import os.path
import random
from tqdm.contrib import tzip
import numpy as np
import soundfile as sf

from mymodule import my_func, const

def random_crop(noise, target_length):
    if len(noise) <= target_length:
        # 長さが足りない場合はループして埋める
        repeat_times = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeat_times)
    start = np.random.randint(0, len(noise) - target_length + 1)
    return noise[start:start + target_length]

def mix_snr(speech, noise, snr_db):
    # SNR調整
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-10))
    return speech + noise

def load_wav(filepath):
    data, sr = sf.read(filepath)
    return data, sr

def save_wav(filepath, data, sr):
    sf.write(filepath, data, sr)

def main(speech_dir,noise_dir, out_dir, snr=5):
    speech_list = my_func.get_file_list(speech_dir)
    noise_list = my_func.get_file_list(noise_dir)

    noise_list = [random.choice(noise_list) for _ in range(len(speech_list))]

    for speech_file, noise_file in tzip(speech_list, noise_list):
        speech, sr = load_wav(speech_file)
        noise, _ = load_wav(noise_file)
        noise = random_crop(noise, len(speech))

        mixed = mix_snr(speech, noise, snr_db=snr)
        out_name = f"{my_func.get_file_name(speech_file)[0]}_{my_func.get_file_name(noise_file)[0]}_{int(snr):03}dB.wav"
        out_path = os.path.join(out_dir, out_name)
        # print(type(sr))
        save_wav(out_path, mixed, sr)

if __name__ == '__main__':

    for train_test in ['train', 'test']:
        speech_dir = f"{const.SAMPLE_DATA_DIR}/speech/JA/{train_test}"
        noise_dir = f"{const.SAMPLE_DATA_DIR}/noise/hoth.wav"
        out_dir = f"{const.MIX_DATA_DIR}/GNN/JA_hoth_5dB/{train_test}/noise_only"
        my_func.make_dir(out_dir)
        main(speech_dir,noise_dir, out_dir, snr=5)

    # mac_sound_dir = "/Users/a/Documents/sound_data/"
    # speech_dir = f"{mac_sound_dir}/sample_data/speech/subset_DEMAND/{train_test}"
    # noise_dir = f"{mac_sound_dir}/sample_data/noise/hoth.wav"
    # out_dir = f"{mac_sound_dir}/mix_data/GNN/subset_DEMAND_hoth_5dB/{train_test}"
    # my_func.make_dir(out_dir)
    # main(speech_dir,noise_dir, out_dir, snr=5)
