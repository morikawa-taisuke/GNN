import os
import wave
import csv

import my_func


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        params = wav_file.getparams()
        frames = wav_file.getnframes()
        rate = params.framerate
        duration = frames / float(rate)
    return duration


def write_durations_to_csv(input_dir, output_csv):
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Filename', 'Duration (sec)'])

        for wav_file in wav_files:
            file_path = os.path.join(input_dir, wav_file)
            duration = get_wav_duration(file_path)
            csv_writer.writerow([wav_file, duration])
            # print(f"{wav_file}, {duration}")


if __name__ == "__main__":
    dir_name = 'C:\\Users\\kataoka-lab\\Desktop\\sound_data\\mix_data\\sebset_DEMAND_hoth_1010dB_05sec_4ch\\train'
    subdir_list = my_func.get_subdir_list(dir_name)
    for subdir in subdir_list:
        input_dir = os.path.join(dir_name, subdir)  # 入力ディレクトリ
        output_csv = f'{subdir}.csv'  # 出力CSVファイル
        write_durations_to_csv(input_dir, output_csv)
