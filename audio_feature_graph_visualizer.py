import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph


class AudioGraphVisualizer:
    def __init__(
        self, input_channels=1, output_channels=512, kernel_size=64, stride=32
    ):
        self.conv1d = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def create_knn_graph(self, x_nodes_batched, k, batch_size, num_nodes_per_sample):
        """k-NNグラフを作成"""
        batch_indices = torch.arange(batch_size).repeat_interleave(num_nodes_per_sample)
        edge_index = knn_graph(x=x_nodes_batched, k=k, batch=batch_indices, loop=False)
        return edge_index

    def process_and_visualize(self, audio_input, k=8):
        """既存のメソッドを修正"""
        # 1. 1D Convolution
        x_conv = self.conv1d(audio_input)

        # 2. データの整形
        batch_size, channels, length = x_conv.size()
        x_nodes = x_conv.permute(0, 2, 1).reshape(-1, channels)

        # 3. k-NNグラフの作成
        edge_index = self.create_knn_graph(x_nodes, k, batch_size, length)

        # 4. ノードの特徴量をCSVとして保存
        nodes_df = pd.DataFrame(x_nodes.detach().cpu().numpy())
        nodes_df.columns = [f"feature_{i}" for i in range(nodes_df.shape[1])]
        nodes_df.to_csv("nodes.csv", index=True)

        # 5. エッジの隣接行列をCSVとして保存
        self.save_adjacency_matrix(x_nodes, edge_index, length)

        # 統計情報の表示
        print("\nグラフ統計:")
        print(f"ノード数: {length}")
        print(f"エッジ数: {edge_index.shape[1]}")
        print(f"1ノードあたりの平均エッジ数: {edge_index.shape[1] / length:.2f}")

        return x_conv, edge_index

    def visualize_graph(self, x_nodes, edge_index, time_length, max_nodes=100):
        """グラフの可視化"""
        # グラフオブジェクトの作成
        G = nx.Graph()

        # ノードの追加（表示制限のため最初の max_nodes 個のみ）
        num_nodes = min(len(x_nodes), max_nodes)
        pos = {}

        for i in range(num_nodes):
            G.add_node(i)
            # ノードの位置を時系列に沿って配置
            pos[i] = (i % time_length, i // time_length)

        # エッジの追加
        edges = edge_index.t().cpu().numpy()
        for src, dst in edges:
            if src < num_nodes and dst < num_nodes:
                G.add_edge(src, dst)

        # 描画
        plt.figure(figsize=(100, 8))
        # nx.draw(G, pos,
        #         node_color='lightblue',
        #         node_size=100,
        #         edge_color='gray',
        #         width=0.5,
        #         with_labels=False)
        # ノードの描画
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=100)

        # エッジの描画（曲線）
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="gray",
            width=0.5,
            connectionstyle="arc3,rad=0.2",
            arrows=True,
        )  # rad パラメータで曲がり具合を調整

        plt.title("Audio Feature Graph Visualization")
        plt.show()

    def save_graph_data(self, x_nodes, edge_index):
        """ノードとエッジの情報をCSVとして保存"""
        # ノードデータの保存
        nodes_df = pd.DataFrame(x_nodes.detach().cpu().numpy())
        nodes_df.to_csv(
            "nodes.csv",
            index=True,
            header=[f"feature_{i}" for i in range(nodes_df.shape[1])],
        )

        # エッジデータの保存
        edges_df = pd.DataFrame(
            edge_index.t().cpu().numpy(), columns=["source", "target"]
        )
        edges_df.to_csv("edges.csv", index=False)

    def save_adjacency_matrix(
        self, x_nodes, edge_index, length, output_path="edge_matrix.csv"
    ):
        """
        エッジの関係を隣接行列（0-1行列）として保存

        Args:
            x_nodes (torch.Tensor): ノードの特徴量
            edge_index (torch.Tensor): エッジの接続関係 [2, E]
            length (int): 時系列長
            output_path (str): 出力ファイルパス
        """
        # 隣接行列の作成（length × length）の0行列
        adj_matrix = np.zeros((length, length), dtype=np.int8)

        # エッジ情報を隣接行列に変換
        edges = edge_index.t().cpu().numpy()
        for src, dst in edges:
            # length以上のインデックスは無視
            if src < length and dst < length:
                adj_matrix[src, dst] = 1  # エッジが存在する場合は1
                adj_matrix[dst, src] = 1  # 無向グラフなので対称にする

        # CSVとして保存（インデックスと列名を付けない）
        np.savetxt(output_path, adj_matrix, fmt="%d", delimiter=",")
        print(f"隣接行列を {output_path} に保存しました")


def main():
    # パラメータ設定
    input_channels = 1
    output_channels = 512
    kernel_size = 64
    stride = 32
    sample_rate = 16000
    duration = 1  # 1秒

    # サンプルの音声データを生成（実際の使用時は実際の音声データを使用）
    audio_input = torch.randn(1, input_channels, sample_rate * duration)

    # 可視化器の初期化と実行
    visualizer = AudioGraphVisualizer(
        input_channels=input_channels,
        output_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
    )

    # 処理と可視化の実行
    x_conv, edge_index = visualizer.process_and_visualize(audio_input, k=8)

    print(f"入力音声サイズ: {audio_input.shape}")
    print(f"1D Conv出力サイズ: {x_conv.shape}")
    print(f"作成されたエッジ数: {edge_index.shape[1]}")
    print(f"ノードデータは'nodes.csv'に、エッジデータは'edges.csv'に保存されました。")


if __name__ == "__main__":
    main()
