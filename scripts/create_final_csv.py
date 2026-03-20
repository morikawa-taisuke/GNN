import csv
import json
import os.path
import sys
from pathlib import Path

from tqdm import tqdm

# ===================================================================
# 笆ｼ笆ｼ笆ｼ 險ｭ螳夐・岼 笆ｼ笆ｼ笆ｼ
# ===================================================================

# --- 蜈･蜉幄ｨｭ螳・---
# 繧ｵ繝ｳ繝励Μ繝ｳ繧ｰ貂医∩縺ｮJSON繝輔ぃ繧､繝ｫ
DEFAULT_JSON_PATH = "sampled_vctk_file_list.json"

# 諡｡蠑ｵ縺輔ｌ縺溘ョ繝ｼ繧ｿ繧ｻ繝・ヨ縺ｮ隕ｪ繝・ぅ繝ｬ繧ｯ繝医Μ
DEFAULT_DATASET_ROOT = Path("C:/Users/kataoka-lab/Desktop/sound_data/mix_data/DEMAND_DEMAND_5dB_500msec")

# --- 蜃ｺ蜉幄ｨｭ螳・---
# CSV繝輔ぃ繧､繝ｫ縺ｮ繝倥ャ繝繝ｼ
CSV_HEADER = [
    "clean",
    "noise_only",
    "reverb_only",
    "noise_reverb",
]

# 蜷・浹螢ｰ繝輔ぃ繧､繝ｫ縺ｮ遞ｮ鬘槭↓蟇ｾ蠢懊☆繧九ョ繧｣繝ｬ繧ｯ繝医Μ蜷・
CONDITION_DIRS = ["clean", "noise_only", "reverbe_only", "noise_reverb"]

# ===================================================================
# 笆ｲ笆ｲ笆ｲ 險ｭ螳夐・岼 笆ｲ笆ｲ笆ｲ
# ===================================================================


def create_final_csv_list_prefix(json_path: Path, dataset_root: Path):
    """
    繧ｵ繝ｳ繝励Μ繝ｳ繧ｰ貂医∩JSON繧貞渕縺ｫ縲√ヵ繧｡繧､繝ｫ蜷阪・蜑肴婿荳閾ｴ縺ｧ繝・・繧ｿ繧ｻ繝・ヨ繧呈､懃ｴ｢縺励・
    譛邨ら噪縺ｪ繝輔ぃ繧､繝ｫ繝ｪ繧ｹ繝・SV繧剃ｽ懈・縺吶ｋ縲・
    """
    # ---- 1. 蜈･蜉帙ヵ繧｡繧､繝ｫ縺ｮ繝√ぉ繝・け ----
    if not json_path.is_file():
        print(f"笶・繧ｨ繝ｩ繝ｼ: 蜈･蜉妍SON繝輔ぃ繧､繝ｫ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {json_path}", file=sys.stderr)
        sys.exit(1)
    if not dataset_root.is_dir():
        print(f"笶・繧ｨ繝ｩ繝ｼ: 繝・・繧ｿ繧ｻ繝・ヨ縺ｮ繝ｫ繝ｼ繝医ョ繧｣繝ｬ繧ｯ繝医Μ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    print("笨・蜈･蜉帙ヵ繧｡繧､繝ｫ縺ｮ繝√ぉ繝・け螳御ｺ・・)
    print(f"当 JSON蜈･蜉・ {json_path}")
    print(f"朕 繝・・繧ｿ繧ｻ繝・ヨ繝ｫ繝ｼ繝・ {dataset_root}")

    # ---- 2. JSON繝・・繧ｿ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ ----
    with open(json_path, "r", encoding="utf-8") as f:
        all_splits_info = json.load(f)

    # ---- 3. 蜷・そ繝・ヨ・・rain, val, test・峨＃縺ｨ縺ｫCSV繧剃ｽ懈・ ----
    for split_name, speakers_data in all_splits_info.items():
        print(f"\n======== {split_name.upper()} 繧ｻ繝・ヨ縺ｮ蜃ｦ逅・ｒ髢句ｧ・========")

        output_csv_path = os.path.join(dataset_root, F"{split_name}.csv")
        all_rows = []
        missing_files_count = 0

        # JSON縺九ｉ蜃ｦ逅・ｯｾ雎｡縺ｮ繝輔ぃ繧､繝ｫ蜷阪Μ繧ｹ繝医ｒ菴懈・
        file_list = []
        for _, data in speakers_data.items():
            file_list.extend(data["filenames"])

        # tqdm繧剃ｽｿ縺｣縺ｦ騾ｲ謐励ヰ繝ｼ繧定｡ｨ遉ｺ
        for base_filename in tqdm(file_list, desc=f"Processing {split_name}"):
            row_paths = []

            # 4遞ｮ鬘槭・髻ｳ螢ｰ繝輔ぃ繧､繝ｫ縺ｮ繝代せ繧呈ｧ狗ｯ・
            for condition in CONDITION_DIRS:
                # 笘・・笘・螟画峩轤ｹ 笘・・笘・
                # 讀懃ｴ｢蟇ｾ雎｡縺ｮ繝・ぅ繝ｬ繧ｯ繝医Μ繝代せ
                search_dir = dataset_root / split_name / condition
                # 蜑肴婿荳閾ｴ縺ｧ繝輔ぃ繧､繝ｫ繧呈､懃ｴ｢縺吶ｋ縺溘ａ縺ｮ繝代ち繝ｼ繝ｳ
                glob_pattern = f"{base_filename}*.wav"

                # 繝代ち繝ｼ繝ｳ縺ｫ荳閾ｴ縺吶ｋ繝輔ぃ繧､繝ｫ縺ｮ繝ｪ繧ｹ繝医ｒ蜿門ｾ・
                # .glob()縺ｯ繧ｸ繧ｧ繝阪Ξ繝ｼ繧ｿ繧定ｿ斐☆縺溘ａ縲√Μ繧ｹ繝医↓螟画鋤
                found_files = list(search_dir.glob(glob_pattern))

                # 繝輔ぃ繧､繝ｫ縺瑚ｦ九▽縺九ｌ縺ｰ譛蛻昴・繝輔ぃ繧､繝ｫ縺ｮ邨ｶ蟇ｾ繝代せ繧偵√↑縺代ｌ縺ｰ遨ｺ譁・ｭ励ｒ霑ｽ蜉
                if found_files:
                    row_paths.append(str(found_files[0].resolve()))
                else:
                    row_paths.append("")
                    missing_files_count += 1

            all_rows.append(row_paths)

        # ---- 4. CSV繝輔ぃ繧､繝ｫ縺ｸ縺ｮ譖ｸ縺崎ｾｼ縺ｿ ----
        if not all_rows:
            print(f"  - '{split_name}' 繧ｻ繝・ヨ縺ｫ縺ｯ蜃ｦ逅・☆繧九ョ繝ｼ繧ｿ縺後≠繧翫∪縺帙ｓ縺ｧ縺励◆縲・)
            continue

        try:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
                writer.writerows(all_rows)

            print(f"笨・'{output_csv_path}' 縺梧ｭ｣蟶ｸ縺ｫ菴懈・縺輔ｌ縺ｾ縺励◆縲らｷ上Ξ繧ｳ繝ｼ繝画焚: {len(all_rows)}")
            if missing_files_count > 0:
                print(
                    f"笞・・ 豕ｨ諢・ {missing_files_count} 蛟九・繝輔ぃ繧､繝ｫ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ縺ｧ縺励◆縲・SV蜀・・遨ｺ谺・ｒ遒ｺ隱阪＠縺ｦ縺上□縺輔＞縲・
                )

        except IOError as e:
            print(f"笶・繧ｨ繝ｩ繝ｼ: '{output_csv_path}' 縺ｮ譖ｸ縺崎ｾｼ縺ｿ縺ｫ螟ｱ謨励＠縺ｾ縺励◆: {e}", file=sys.stderr)

    print("\n脂 蜈ｨ縺ｦ縺ｮ蜃ｦ逅・′螳御ｺ・＠縺ｾ縺励◆縲・)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="JSON縺ｨ繝・・繧ｿ繧ｻ繝・ヨ繝・ぅ繝ｬ繧ｯ繝医Μ縺九ｉ蜑肴婿荳閾ｴ縺ｧ讀懃ｴ｢縺励∵怙邨ら噪縺ｪCSV繝輔ぃ繧､繝ｫ繝ｪ繧ｹ繝医ｒ菴懈・縺励∪縺吶・
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"蜈･蜉帙→縺ｪ繧九し繝ｳ繝励Μ繝ｳ繧ｰ貂医∩JSON繝輔ぃ繧､繝ｫ縺ｮ繝代せ (繝・ヵ繧ｩ繝ｫ繝・ {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"繝・・繧ｿ繧ｻ繝・ヨ縺ｮ隕ｪ繝・ぅ繝ｬ繧ｯ繝医Μ縺ｮ繝代せ (繝・ヵ繧ｩ繝ｫ繝・ {DEFAULT_DATASET_ROOT})",
    )

    args = parser.parse_args()

    create_final_csv_list_prefix(json_path=args.json_path, dataset_root=args.dataset_root)
