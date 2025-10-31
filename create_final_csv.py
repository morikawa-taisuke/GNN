import csv
import json
import os.path
import sys
from pathlib import Path

from tqdm import tqdm

# ===================================================================
# ▼▼▼ 設定項目 ▼▼▼
# ===================================================================

# --- 入力設定 ---
# サンプリング済みのJSONファイル
DEFAULT_JSON_PATH = "sampled_vctk_file_list.json"

# 拡張されたデータセットの親ディレクトリ
DEFAULT_DATASET_ROOT = Path("C:/Users/kataoka-lab/Desktop/sound_data/mix_data/DEMAND_DEMAND_5dB_500msec")

# --- 出力設定 ---
# CSVファイルのヘッダー
CSV_HEADER = [
    "clean",
    "noise_only",
    "reverb_only",
    "noise_reverb",
]

# 各音声ファイルの種類に対応するディレクトリ名
CONDITION_DIRS = ["clean", "noise_only", "reverbe_only", "noise_reverbe"]

# ===================================================================
# ▲▲▲ 設定項目 ▲▲▲
# ===================================================================


def create_final_csv_list_prefix(json_path: Path, dataset_root: Path):
    """
    サンプリング済みJSONを基に、ファイル名の前方一致でデータセットを検索し、
    最終的なファイルリストCSVを作成する。
    """
    # ---- 1. 入力ファイルのチェック ----
    if not json_path.is_file():
        print(f"❌ エラー: 入力JSONファイルが見つかりません: {json_path}", file=sys.stderr)
        sys.exit(1)
    if not dataset_root.is_dir():
        print(f"❌ エラー: データセットのルートディレクトリが見つかりません: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    print("✅ 入力ファイルのチェック完了。")
    print(f"📖 JSON入力: {json_path}")
    print(f"💽 データセットルート: {dataset_root}")

    # ---- 2. JSONデータの読み込み ----
    with open(json_path, "r", encoding="utf-8") as f:
        all_splits_info = json.load(f)

    # ---- 3. 各セット（train, val, test）ごとにCSVを作成 ----
    for split_name, speakers_data in all_splits_info.items():
        print(f"\n======== {split_name.upper()} セットの処理を開始 ========")

        output_csv_path = os.path.join(dataset_root, F"{split_name}.csv")
        all_rows = []
        missing_files_count = 0

        # JSONから処理対象のファイル名リストを作成
        file_list = []
        for _, data in speakers_data.items():
            file_list.extend(data["filenames"])

        # tqdmを使って進捗バーを表示
        for base_filename in tqdm(file_list, desc=f"Processing {split_name}"):
            row_paths = []

            # 4種類の音声ファイルのパスを構築
            for condition in CONDITION_DIRS:
                # ★★★ 変更点 ★★★
                # 検索対象のディレクトリパス
                search_dir = dataset_root / split_name / condition
                # 前方一致でファイルを検索するためのパターン
                glob_pattern = f"{base_filename}*.wav"

                # パターンに一致するファイルのリストを取得
                # .glob()はジェネレータを返すため、リストに変換
                found_files = list(search_dir.glob(glob_pattern))

                # ファイルが見つかれば最初のファイルの絶対パスを、なければ空文字を追加
                if found_files:
                    row_paths.append(str(found_files[0].resolve()))
                else:
                    row_paths.append("")
                    missing_files_count += 1

            all_rows.append(row_paths)

        # ---- 4. CSVファイルへの書き込み ----
        if not all_rows:
            print(f"  - '{split_name}' セットには処理するデータがありませんでした。")
            continue

        try:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
                writer.writerows(all_rows)

            print(f"✅ '{output_csv_path}' が正常に作成されました。総レコード数: {len(all_rows)}")
            if missing_files_count > 0:
                print(
                    f"⚠️  注意: {missing_files_count} 個のファイルが見つかりませんでした。CSV内の空欄を確認してください。"
                )

        except IOError as e:
            print(f"❌ エラー: '{output_csv_path}' の書き込みに失敗しました: {e}", file=sys.stderr)

    print("\n🎉 全ての処理が完了しました。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="JSONとデータセットディレクトリから前方一致で検索し、最終的なCSVファイルリストを作成します。"
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"入力となるサンプリング済みJSONファイルのパス (デフォルト: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"データセットの親ディレクトリのパス (デフォルト: {DEFAULT_DATASET_ROOT})",
    )

    args = parser.parse_args()

    create_final_csv_list_prefix(json_path=args.json_path, dataset_root=args.dataset_root)
