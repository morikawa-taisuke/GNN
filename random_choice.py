import json
import random
import sys

# --- 設定項目 ---

# 読み込むJSONファイル
# （ファイル名を変更した場合はここを修正してください）
INPUT_JSON_FILE = "vctk_split_file_list_filenames.json"

# 出力する新しいJSONファイルの名前
OUTPUT_JSON_FILE = "sampled_vctk_file_list.json"

# 各話者からサンプリングするファイルの割合
SAMPLING_RATIO = 2 / 3


def create_sampled_json(json_path: str, output_path: str, ratio: float):
    """
    JSONファイルからファイルリストを読み込み、指定された割合でランダムサンプリングし、
    実験用の新しいJSONファイルを生成する。

    Args:
        json_path (str): 入力JSONファイル名。
        output_path (str): 出力JSONファイル名。
        ratio (float): 0から1の間のサンプリング割合。
    """
    # 1. JSONファイルの読み込み
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_splits_info = json.load(f)
        print(f"✅ JSONファイル '{json_path}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"❌ エラー: 入力ファイル '{json_path}' が見つかりません。", file=sys.stderr)
        print("前のステップのプログラムを先に実行してください。", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ エラー: '{json_path}' は有効なJSONファイルではありません。", file=sys.stderr)
        sys.exit(1)

    # 2. 新しいJSONファイルのための辞書を準備
    sampled_splits_info = {}

    # 3. 各セット（train, val, test）ごとに処理
    for split_name, speakers_data in all_splits_info.items():
        print(f"\n🔄 '{split_name}' セットの処理を開始...")

        # 新しい辞書にセット名を追加
        sampled_splits_info[split_name] = {}

        # 各話者ごとにサンプリング処理
        # sorted() を使って処理順を固定し、再現性を高める
        for speaker_id, data in sorted(speakers_data.items()):
            all_filenames = data["filenames"]

            # サンプリングするファイル数を計算
            num_to_sample = int(len(all_filenames) * ratio)

            # リストからランダムにサンプリング
            # random.sampleは重複なく要素を選択する
            sampled_filenames = sorted(random.sample(all_filenames, k=num_to_sample))

            # サンプリング後のデータを新しい辞書に格納
            sampled_splits_info[split_name][speaker_id] = {
                "file_count": len(sampled_filenames),
                "filenames": sampled_filenames,
            }

            print(
                f"  - 話者 {speaker_id}: {len(all_filenames)} ファイルから {len(sampled_filenames)} ファイルをサンプリングしました。"
            )

    # 4. 新しいJSONファイルへの書き込み
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # indent=4 で人間が読みやすいように整形して出力
            json.dump(sampled_splits_info, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 新しいJSONファイル '{output_path}' が正常に作成されました。")
    except IOError as e:
        print(f"❌ エラー: '{output_path}' の書き込みに失敗しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    create_sampled_json(INPUT_JSON_FILE, OUTPUT_JSON_FILE, SAMPLING_RATIO)
