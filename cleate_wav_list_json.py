import json
import sys
from pathlib import Path

# ===================================================================
# ▼▼▼ 修正箇所 ▼▼▼
# 'train', 'val', 'test' フォルダが含まれる親ディレクトリのパスを指定してください
DATASET_PARENT_PATH = Path("C:/Users/kataoka-lab/Desktop/sound_data/sample_data/speech/speeker_DEMAND")
# ▲▲▲ 修正箇所 ▲▲▲
# ===================================================================

# 出力するJSONファイルの名前
OUTPUT_JSON_FILE = "./vctk_split_file_list_filenames.json"


def create_split_dataset_json(parent_path: Path, output_file: str):
    """
    train/val/testに分割されたデータセット構造をスキャンし、
    話者ごとのファイル名（拡張子なし）情報をJSONに出力する。

    Args:
        parent_path (Path): 'train', 'val', 'test'を含む親ディレクトリのパス。
        output_file (str): 出力するJSONファイル名。
    """
    # 指定されたパスが存在するかチェック
    if not parent_path.is_dir():
        print(f"エラー: 指定されたパスが見つかりません: {parent_path}", file=sys.stderr)
        print("プログラムの 'DATASET_PARENT_PATH' を正しいパスに修正してください。", file=sys.stderr)
        sys.exit(1)

    print(f"'{parent_path}' をスキャンしています...")

    # 全ての分割セット（train/val/test）の情報を格納するメインの辞書
    all_splits_info = {}
    split_names = ["train", "val", "test"]

    # 各セット（train, val, test）に対して処理を実行
    for split_name in split_names:
        split_path = parent_path / split_name
        if not split_path.is_dir():
            print(f"  - '{split_name}' ディレクトリが見つからないため、スキップします。")
            continue

        print(f"  - '{split_name}' セットを処理中...")
        all_splits_info[split_name] = {}

        # 話者ディレクトリをループ処理
        speaker_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])

        for speaker_dir in speaker_dirs:
            speaker_id = speaker_dir.name
            # 'p' で始まるディレクトリを話者ディレクトリとみなす
            if not speaker_id.startswith("p"):
                continue

            # 音声ファイル (.wav) のリストを取得
            audio_files = list(speaker_dir.glob("*.wav"))
            if not audio_files:
                continue

            # ★★★ 変更点 ★★★
            # 各ファイルの「ファイル名（拡張子なし）」をリストに格納
            # pathlib.Pathオブジェクトの .stem プロパティを使用
            filenames = sorted([f.stem for f in audio_files])
            file_count = len(filenames)

            # 辞書に情報を格納 (キーを 'filenames' に変更)
            all_splits_info[split_name][speaker_id] = {
                "file_count": file_count,
                "filenames": filenames,
            }
            print(f"    - 話者 {speaker_id} を発見。音声ファイル数: {file_count}")

    # 最終的な辞書をJSONファイルに書き出す
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_splits_info, f, indent=4, ensure_ascii=False)
        print(f"\n成功！'{output_file}' にファイルリストが作成されました。")
    except IOError as e:
        print(f"\nエラー: ファイルの書き込みに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    create_split_dataset_json(DATASET_PARENT_PATH, OUTPUT_JSON_FILE)
