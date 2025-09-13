import os

import matplotlib.pyplot as plt
import networkx as nx
# 追加のインポート
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torchinfo import summary

from models.graph_utils import GraphBuilder, GraphConfig, NodeSelectionType, EdgeSelectionType

from mymodule import confirmation_GPU

# PyTorchのCUDAメモリ管理設定。セグメントを拡張可能にすることで、断片化によるメモリ不足エラーを緩和します。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CUDAが利用可能かチェックし、利用可能ならGPUを、そうでなければCPUを使用するデバイスとして設定します。
device = confirmation_GPU.get_device()
print(f"GNN_encoder.py 使用デバイス: {device}")


def visualize_spectral_graph(x_nodes, edge_index, freq_bins, time_frames, max_time_frames=30):
    """
    スペクトログラムの格子構造を保持したGNNのグラフ構造を可視化します。

    Args:
        x_nodes (torch.Tensor): ノードの特徴量 [N, C]
        edge_index (torch.Tensor): エッジの接続関係 [2, E]
        freq_bins (int): 周波数ビンの数
        time_frames (int): 時間フレームの数
        max_time_frames (int): 表示する最大時間フレーム数
    """
    # 表示する時間フレームを制限
    time_frames = min(time_frames, max_time_frames)

    # グラフの作成
    G = nx.Graph()

    # ノードの位置を格子状に配置
    pos = {}
    node_labels = {}

    # ノードの追加と位置の設定
    for t in range(time_frames):
        for f in range(freq_bins):
            node_idx = t * freq_bins + f
            G.add_node(node_idx)
            # 位置を設定（x座標が時間、y座標が周波数）
            pos[node_idx] = (t, f)
            node_labels[node_idx] = f"{node_idx}"

    # エッジの追加（表示範囲内のエッジのみ）
    max_node_idx = time_frames * freq_bins
    mask = (edge_index[0] < max_node_idx) & (edge_index[1] < max_node_idx)
    edge_index_subset = edge_index[:, mask]
    edges = edge_index_subset.t().cpu().numpy()
    G.add_edges_from(edges)

    # グラフの描画
    plt.figure(figsize=(15, 10))

    # ノードの描画
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=100)

    # エッジの描画
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

    # ラベルの描画（オプション）
    # nx.draw_networkx_labels(G, pos, node_labels, font_size=8)

    plt.title(f"{freq_bins}x{time_frames}")
    plt.xlabel("time")
    plt.ylabel("Frequency")

    # グリッドの表示
    plt.grid(True, linestyle="--", alpha=0.3)

    # 軸の範囲を設定
    plt.xlim(-1, time_frames)
    plt.ylim(-1, freq_bins)

    plt.show()

    # 接続統計の表示
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"\nグラフ統計:")
    print(f"総ノード数: {len(G.nodes())}")
    print(f"総エッジ数: {len(G.edges())}")
    print(f"平均次数: {np.mean(degrees):.2f}")
    print(f"最小次数: {np.min(degrees)}")
    print(f"最大次数: {np.max(degrees)}")

# --- 既存のユーティリティークラス (SpeqGNN.pyから再利用) ---
class DoubleConv(nn.Module):
    """(畳み込み => バッチ正規化 => ReLU) を2回繰り返すブロック"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """順伝播"""
        return self.double_conv(x)


class Down(nn.Module):
    """ダウンサンプリングブロック (マックスプーリング + DoubleConv)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool1d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        """順伝播"""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """アップサンプリングブロック"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # パディングの計算
        diff = x2.size(2) - x1.size(2)
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        # 特徴量の結合
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GCN(nn.Module):
    """3層のGraph Convolutional Network (GCN) モデル"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """順伝播"""
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class GAT(nn.Module):
    """3層のGraph Attention Network (GAT) モデル"""

    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout_rate=0.5):
        super(GAT, self).__init__()
        self.heads = heads
        self.dropout_rate = dropout_rate
        # 1層目: マルチヘッドアテンションを適用
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
        # 2層目: 入力は1層目のヘッドを結合したものになる (hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_rate)
        # 3層目: 出力層。ヘッドのアベレージングを行うため concat=False に設定
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout_rate)

    def forward(self, x, edge_index):
        """順伝播"""
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class GNNEncoder(nn.Module):
    """
    GNNをエンコーダとして、その後にU-Netを実装した音声強調/分離モデル。
    """

    def __init__(
        self,
        n_channels,
        n_classes=1,
        hidden_dim_gnn=32,
        num_node=8,
        gnn_type="GCN",
        gnn_heads=4,
        gnn_dropout=0.6,
        graph_config=None,
    ):
        """
        Args:
            n_channels (int): 入力音声波形のチャネル数 (例: モノラルなら1)。
            n_classes (int): 出力マスクのチャネル数 (通常は1)。
            hidden_dim_gnn (int): GNNの隠れ層の次元数。
            num_node (int): k-NNグラフを作成する際の近傍ノード数。
            gnn_type (str): 使用するGNNの種類 ('GCN' または 'GAT')。
            gnn_heads (int): GATの場合のマルチヘッドアテンションのヘッド数。
            gnn_dropout (float): GATの場合のドロップアウト率。
        """
        super(GNNEncoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_node_gnn = num_node
        self.gnn_type = gnn_type

        # 1D畳み込みエンコーダ (波形 -> 潜在特徴量)
        self.encoder_dim_1d_conv = 512
        self.sampling_rate = 16000
        self.win = 4  # ms (ウィンドウ長)
        self.win_samples = int(self.sampling_rate * self.win / 1000)
        self.stride_samples = self.win_samples // 2

        self.initial_encoder = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=self.encoder_dim_1d_conv,
            kernel_size=self.win_samples,
            bias=False,
            stride=self.stride_samples,
        )

        # GNNエンコーダ (潜在特徴量に対して動作)
        # GNNの入出力次元は、initial_encoderの出力次元とU-Netの入力次元に合わせる
        if gnn_type == "GCN":
            self.gnn_encoder = GCN(self.encoder_dim_1d_conv, hidden_dim_gnn, self.encoder_dim_1d_conv)
        elif gnn_type == "GAT":
            self.gnn_encoder = GAT(
                self.encoder_dim_1d_conv,
                hidden_dim_gnn,
                self.encoder_dim_1d_conv,
                heads=gnn_heads,
                dropout_rate=gnn_dropout,
            )
        else:
            raise ValueError(f"サポートされていないGNNタイプです: {gnn_type}。'GCN' または 'GAT' を選択してください。")

        # U-Netエンコーダパス
        # GNNの出力は [B, C_feat, L_encoded] となり、これをunsqueezeしてU-Netの2DConvの入力とする
        # したがって、incのin_channelsは1となる
        self.inc = DoubleConv(512, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # U-Netデコーダパス
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        # マスク用の出力畳み込み
        self.outc = nn.Sequential(nn.Conv1d(64, n_classes, kernel_size=1), nn.Sigmoid())

        # 最終的な1D転置畳み込みデコーダ (マスクされた潜在特徴量 -> 波形)
        self.final_decoder = nn.ConvTranspose1d(
            in_channels=self.encoder_dim_1d_conv,
            out_channels=self.n_channels,
            kernel_size=self.win_samples,
            bias=False,
            stride=self.stride_samples,
        )
        # デフォルトのグラフ設定
        if graph_config is None:
            graph_config = GraphConfig(
                num_edges=num_node,
                node_selection=NodeSelectionType.ALL,
                edge_selection=EdgeSelectionType.KNN,
                bidirectional=True,
            )
        self.graph_builder = GraphBuilder(graph_config)

    def forward(self, x_waveform):
        """
        順伝播
        Args:
            x_waveform (torch.Tensor): 入力音声波形 [バッチ, チャネル数, 時間長] または [バッチ, 時間長]
        """
        # 入力波形が3Dであることを確認する: [B, C, L]
        if x_waveform.dim() == 2:
            x_waveform = x_waveform.unsqueeze(1)  # チャンネル次元がない場合は追加

        # 最終的な出力をトリミングするために、元の波形の長さを保存
        original_waveform_length = x_waveform.size(-1)

        # 1. 初期1D畳み込みエンコーダ (波形 -> 潜在特徴量)
        # x_encoded: [B, encoder_dim_1d_conv, L_encoded]
        x_encoded = self.initial_encoder(x_waveform)

        batch_size, feature_dim_encoded, length_encoded = x_encoded.size()

        # GNN用にリシェイプ: [B, C, L_encoded] -> [B * L_encoded, C]
        # 各時間フレームがノードとなり、その特徴量はencoder_dim_1d_conv次元
        x_nodes = x_encoded.permute(0, 2, 1).reshape(-1, feature_dim_encoded)

        # GNN用のグラフを作成
        num_nodes_per_sample = length_encoded
        if num_nodes_per_sample > 0:
            edge_index = self.graph_builder.create_batch_graph(x_nodes, batch_size, length_encoded)

        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x_nodes.device)

        # グラフの可視化（デバッグ時のみ実行）
        # visualize_spectral_graph(
        #     x_nodes,
        #     edge_index,
        #     freq_bins=feature_dim_encoded,
        #     time_frames=length_encoded,
        # )

        # 2. GNNエンコーダ (GNNを使用してノードを処理)
        x_gnn_out_flat = self.gnn_encoder(x_nodes, edge_index)  # [B * L_encoded, feature_dim_encoded]

        # GNNの出力をU-Net用の特徴マップ形式にリシェイプ
        # [B * L_encoded, C] -> [B, L_encoded, C] -> [B, C, L_encoded]
        x_gnn_out_reshaped = x_gnn_out_flat.view(batch_size, length_encoded, feature_dim_encoded).permute(0, 2, 1)

        # U-Netへの入力: [B, 1, C, L_encoded] (Cはfeature_dim_encoded)
        # x_unet_input = x_gnn_out_reshaped.unsqueeze(1)

        # 3. U-Netエンコーダパス
        x1 = self.inc(x_gnn_out_reshaped)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # ボトルネック特徴マップ

        # 4. U-Netデコーダパス
        d3 = self.up1(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        # 5. マスクの生成
        mask_pred_raw = self.outc(d1)  # [B, n_classes, H_mask, W_mask]

        # マスクをx_gnn_out_reshapedの次元(feature_dim_encoded, length_encoded)に合うようにリサイズ
        mask_target_H = feature_dim_encoded
        mask_target_W = length_encoded

        if mask_pred_raw.size(2) != length_encoded:
            mask_resized = F.interpolate(
                mask_pred_raw,
                size=length_encoded,
                mode="linear",  # 1次元データには'linear'を使用
                align_corners=False,
            )

        else:
            mask_resized = mask_pred_raw

        # GNNエンコーダの出力(x_gnn_out_reshaped)にマスクを適用
        # x_gnn_out_reshaped: [B, C, L_encoded]
        # mask_resized: [B, n_classes, C, L_encoded]
        if self.n_classes == 1:
            masked_features = x_gnn_out_reshaped * mask_resized.squeeze(1)  # 特徴量全体にマスクをブロードキャスト
        else:
            # 複数クラスを意図する場合、より具体的なロジックが必要です。
            # 一般的な音声強調では、n_classes=1 (単一マスク)です。
            print(f"警告: GNNEncoder - n_classes > 1 ({self.n_classes})。最初のマスクチャネルを使用します。")
            masked_features = x_gnn_out_reshaped * mask_resized[:, 0, :, :]

        # 6. 最終的な1D畳み込みデコーダ (マスクされた潜在特徴量 -> 波形)
        output_waveform = self.final_decoder(masked_features)

        # 出力波形を元の入力長に合わせるためにトリミング
        # これにより、初期エンコーダ/デコーダのストライドによって導入される可能性のあるパディングを処理します。
        output_waveform = output_waveform[:, :, :original_waveform_length]

        return output_waveform


def print_model_summary(model, batch_size, channels, length):
    """モデルのサマリーを表示するヘルパー関数"""
    x = torch.randn(batch_size, channels, length).to(device)
    print(f"\n{model.__class__.__name__} モデルサマリー:")
    summary(model, input_data=x)


if __name__ == "__main__":
    print("GNN_encoder.py のメイン実行（モデルテスト用）")

    # モデルのパラメータ設定
    batch = 1  # バッチサイズ
    num_mic = 1  # マイクの数 (入力チャンネル数)
    length = 16000 * 3  # 音声の長さ (サンプル数): 3秒

    # GNNEncoderUNetモデルのインスタンス化 (GCNタイプ)
    print("\n--- GNNEncoder (GCNをエンコーダとして使用) ---")
    gnn_encoder_unet_gcn = GNNEncoder(
        n_channels=num_mic, n_classes=1, hidden_dim_gnn=32, num_node=8, gnn_type="GCN"
    ).to(device)
    print_model_summary(gnn_encoder_unet_gcn, batch, num_mic, length)

    # GNNEncoderUNetモデルのインスタンス化 (GATタイプ)
    print("\n--- GNNEncoder (GATをエンコーダとして使用) ---")
    gnn_encoder_unet_gat = GNNEncoder(
        n_channels=num_mic,
        n_classes=1,
        hidden_dim_gnn=32,
        num_node=8,
        gnn_type="GAT",
        gnn_heads=4,
        gnn_dropout=0.6,
    ).to(device)
    print_model_summary(gnn_encoder_unet_gat, batch, num_mic, length)

    # サンプル入力データを作成
    dummy_input = torch.randn(batch, num_mic, length).to(device)

    print("\n--- 順伝播の例 (GCNEncoderUNet) ---")
    with torch.no_grad():
        output_gcn = gnn_encoder_unet_gcn(dummy_input)
    print(f"入力形状: {dummy_input.shape}")
    print(f"出力形状: {output_gcn.shape}")

    print("\n--- 順伝播の例 (GATEncoderUNet) ---")
    with torch.no_grad():
        output_gat = gnn_encoder_unet_gat(dummy_input)
    print(f"入力形状: {dummy_input.shape}")
    print(f"出力形状: {output_gat.shape}")

    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPUメモリ使用量 (初期化および順伝播後):")
        print(f"確保済み: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"キャッシュ: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
