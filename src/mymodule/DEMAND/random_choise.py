import os
import random
import shutil


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

def select_random_files(source_dir, num_files, output_file):
    # フォルダが存在するか確認
    if not os.path.isdir(source_dir):
        print(f"{source_dir} is not a valid directory")
        return

    # フォルダ内のすべてのファイルを取得
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 指定された数のファイルをランダムに選択
    selected_files = random.sample(files, min(num_files, len(files)))

    # 選択されたファイルの絶対パスを取得
    selected_files_paths = [os.path.abspath(os.path.join(source_dir, f)) for f in selected_files]

    # 結果を出力ファイルに書き込む
    with open(output_file, 'w') as f:
        for file_path in selected_files_paths:
            f.write(file_path + '\n')

    print(f"{num_files} random files have been selected and written to {output_file}")

def move_files_from_list(file_list_path, target_dir):
    # ファイルリストが存在するか確認
    if not os.path.isfile(file_list_path):
        print(f"{file_list_path} is not a valid file")
        return

    # ターゲットディレクトリが存在するか確認し、存在しない場合は作成
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # ファイルリストを読み込む
    with open(file_list_path, 'r') as f:
        files_to_move = f.read().splitlines()

    # ファイルを移動
    for file_path in files_to_move:
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            target_file_path = os.path.join(target_dir, file_name)
            shutil.move(file_path, target_file_path)
            print(f"Moved {file_path} to {target_file_path}")
        else:
            print(f"{file_path} does not exist or is not a file")


if __name__ == "__main__":
    # 使用例
    """ 指定したフォルダ内から """
    train_test_list = ['train', 'test']
    wave_type_list = ['clean', 'noisy']
    number_of_files = 10

    for wave_type in wave_type_list:
        for train_test in train_test_list:
            serch_dir = f"C:\\Users\\kataoka-lab\\Desktop\\CMGAN\\DEMAND_16kHz\\{train_test}\\{wave_type}\\"
            out_wave_dir = f"C:\\Users\\kataoka-lab\\Desktop\\CMGAN\\sebset_DEMAND\\{train_test}\\"
            source_directory_list = get_subdir_list(serch_dir)
            for source_directory in source_directory_list:
                output_txt_file = f"C:\\Users\\kataoka-lab\\Desktop\\CMGAN\\DEMAND_16kHz\\{train_test}\\{wave_type}\\{source_directory}\\subset_list.txt"
                select_random_files(os.path.join(serch_dir, source_directory), number_of_files, output_txt_file)
                move_files_from_list(output_txt_file, os.path.join(out_wave_dir, wave_type))
