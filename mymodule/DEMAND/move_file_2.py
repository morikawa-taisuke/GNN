import os
import shutil
from collections import defaultdict


def organize_files_by_prefix(source_dir):
    # フォルダが存在するか確認
    if not os.path.isdir(source_dir):
        print(f"{source_dir} is not a valid directory")
        return

    # ファイルをプレフィックスで分類するための辞書
    prefix_dict = defaultdict(list)

    # ファイル名をプレフィックスごとに分類
    for filename in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, filename)):
            prefix = filename.split('_')[0]  # '_'で区切った最初の部分をプレフィックスとする
            prefix_dict[prefix].append(filename)

    # プレフィックスごとに新しいフォルダを作成しファイルを移動
    for prefix, files in prefix_dict.items():
        target_dir = os.path.join(source_dir, prefix)
        os.makedirs(target_dir, exist_ok=True)
        for file in files:
            src_file = os.path.join(source_dir, file)
            dst_file = os.path.join(target_dir, file)
            shutil.move(src_file, dst_file)
            print(f"Moved {src_file} to {dst_file}")


# 使用例
source_directory = "E:\\DEMAND\\noisy\\test"
organize_files_by_prefix(source_directory)
source_directory = "E:\\DEMAND\\noisy\\train"
organize_files_by_prefix(source_directory)
# source_directory = "C:\\Users\\kataoka-lab\\Desktop\\CMGAN\\VCTK-DEMAND_28spk_16kHz\\train\\clean"
# organize_files_by_prefix(source_directory)
