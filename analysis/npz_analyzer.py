import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from mymodule import const
from models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType


class NpzGraphAnalyzer:
    def __init__(self, input_dir, output_dir, sampling_rate=16000, total_stride=256):
        """
        Args:
            input_dir: .npzファイルが格納されているディレクトリ
            output_dir: 解析結果の出力先
            sampling_rate: サンプリング周波数 (GNN.py デフォルト: 16000)
            total_stride: エンコーダ(32) * Downsampling(2^3) = 256
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sec_per_frame = total_stride / sampling_rate # 0.016s

        # ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        print(f"✅ Analyzer initialized.")
        print(f"ℹ️  Time resolution: 1 Frame = {self.sec_per_frame:.3f} sec (16ms)")

        self.stats_list = []

    def run_all(self):
        """ディレクトリ内の全npzファイルを解析"""
        npz_files = list(self.input_dir.glob("*.npz"))
        print(f"Starting analysis for {len(npz_files)} files...")

        for file_path in npz_files:
            self._analyze_single_npz(file_path)

        if self.stats_list:
            self._save_global_stats()
            print(f"Analysis complete. Results saved to {self.output_dir}")
        else:
            print("Error: No data found for analysis.")

    def _analyze_single_npz(self, file_path):
        """単一のnpzファイルの解析"""
        try:
            data = np.load(file_path)
            edge_index = data['edge_index']
            # node_coords は (Batch*Length, 1) の形式で保存されている
            coords = data['node_coords'].flatten()

            # エッジのソースとターゲットの座標取得
            src_idx, dst_idx = edge_index[0], edge_index[1]
            src_t = coords[src_idx]
            dst_t = coords[dst_idx]

            # 距離計算
            diff_t = dst_t - src_t
            dist_t_sec = np.abs(diff_t) * self.sec_per_frame

            # 接続方向統計
            num_edges = len(diff_t)
            future_ratio = np.sum(diff_t > 0) / num_edges
            past_ratio = np.sum(diff_t < 0) / num_edges
            current_ratio = np.sum(diff_t == 0) / num_edges

            stats = {
                "file_name": file_path.name,
                "num_edges": num_edges,
                "avg_time_dist_sec": np.mean(dist_t_sec),
                "max_time_dist_sec": np.max(dist_t_sec),
                "future_ratio": future_ratio,
                "past_ratio": past_ratio,
                "current_ratio": current_ratio
            }
            self.stats_list.append(stats)

        except Exception as e:
            print(f"Error analyzing {file_path.name}: {e}")

    def _save_global_stats(self):
        """全体統計の保存と可視化"""
        df = pd.DataFrame(self.stats_list)
        df.to_csv(self.output_dir / "summary_metrics.csv", index=False)

        # 1. 平均接続距離のヒストグラム
        plt.figure(figsize=(10, 6))
        sns.histplot(df['avg_time_dist_sec'], kde=True, color='skyblue')
        plt.title("Distribution of Average Edge Temporal Length (Seconds)")
        plt.xlabel("Time Distance (sec)")
        plt.ylabel("Count")
        plt.savefig(self.output_dir / "plots" / "avg_distance_hist.png")

        # 2. 接続方向の割合 (Boxplot)
        plt.figure(figsize=(10, 6))
        df_melt = df[['past_ratio', 'current_ratio', 'future_ratio']].melt(
            var_name='Direction', value_name='Ratio'
        )
        sns.boxplot(data=df_melt, x='Direction', y='Ratio', palette="Set2")
        plt.title("Edge Direction Ratios across Dataset")
        plt.savefig(self.output_dir / "plots" / "direction_ratios.png")
        plt.close()

if __name__ == "__main__":
    # --- パス設定 ---
    # GNN.pyの _save_graph_data で保存されるディレクトリを指定してください
    model_list = [
        # "UGCN",
        # "UGAT",
        "SpeqGCN",
        "SpeqGAT",
    ]  # モデルの種類  "UGCN", "UGAT", "ConvTasNet", "UNet"
    wave_types = [
        "noise_only",
        "reverb_only",
        "noise_reverb",
    ]  # 入力信号の種類 (noise_only, reverb_only, noise_reverb)    # UGAT_all_random_reverb_only
    node_selection_list = [
        NodeSelectionType.TEMPORAL,
        NodeSelectionType.ALL
    ]  # ノード選択の方法 (ALL, TEMPORAL)
    edge_selection_list = [
        EdgeSelectionType.RANDOM,
        EdgeSelectionType.KNN
    ]  # エッジ選択の方法 (RANDOM, KNN)
    for model_type in model_list:
        for nose_selection in node_selection_list:
            for edge_selection in edge_selection_list:
                for wave_type in wave_types:
                    INPUT_DIR = f"/Users/a/Documents/sound_data/RESULT/graphs_analysis/Speq_GNN/{model_type}_{nose_selection.value}_{edge_selection.value}_{wave_type}"
                    OUTPUT_DIR = f"./analysis_report/Speq_GNN/{model_type}_{nose_selection.value}_{edge_selection.value}_{wave_type}"

                    analyzer = NpzGraphAnalyzer(INPUT_DIR, OUTPUT_DIR)
                    analyzer.run_all()