import os
import shutil

def organize_files_by_prefix(directory, out_dir):
    # ディレクトリ内のファイルを取得
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # ファイルかどうか確認
        if os.path.isfile(file_path):
            # 先頭4文字を取得
            prefix = filename[:4]

            # 新しいディレクトリのパスを作成
            new_dir = os.path.join(out_dir, prefix)

            # 新しいディレクトリが存在しない場合は作成
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # ファイルを新しいディレクトリに移動
            # shutil.move(file_path, os.path.join(new_dir, filename))
            shutil.copy2(file_path, os.path.join(new_dir, filename))

if __name__ == "__main__":
    # 整理したいディレクトリのパスを指定
    target_directory = "C:\\Users\\kataoka-lab\\Desktop\\sound_data\\sample_data\\speech\\sebset_DEMAND\\test\\clean"
    out_dir = "C:\\Users\\kataoka-lab\\Desktop\\sound_data\\sample_data\\speech\\separate_sebset_DEMAND\\test"

    organize_files_by_prefix(target_directory, out_dir)
