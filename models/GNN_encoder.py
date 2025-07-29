import os

import matplotlib.pyplot as plt
import networkx as nx
# 追加のインポート
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import knn_graph
from torchinfo import summary

# PyTorchのCUDAメモリ管理設定。セグメントを拡張可能にすることで、断片化によるメモリ不足エラーを緩和します。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CUDAが利用可能かチェックし、利用可能ならGPUを、そうでなければCPUを使用するデバイスとして設定します。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GNN_encoder.py using device: {device}")


def visualize_graph_connections(x_nodes, edge_index, max_nodes=50):
    """
    GNNのグラフ構造を可視化します。

    Args:
        x_nodes: ノードの特徴量 [N, C]
        edge_index: エッジの接続関係 [2, E]
        max_nodes: 表示する最大ノード数
    """
    # エッジの情報を表示
    print(f"Total nodes: {x_nodes.size(0)}")
    print(f"Total edges: {edge_index.size(1)}")

    # 最初のmax_nodes個のノードとそれらに関連するエッジだけを表示
    mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
    edge_index_subset = edge_index[:, mask]

    # NetworkXグラフの作成
    G = nx.Graph()

    # ノードの追加
    for i in range(min(max_nodes, x_nodes.size(0))):
        G.add_node(i)

    # エッジの追加
    edges = edge_index_subset.t().cpu().numpy()
    G.add_edges_from(edges)

    # グラフの描画
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
    )
    plt.title(f"Graph visualization (showing first {max_nodes} nodes)")
    plt.show()

    # 接続統計の表示
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"\nAverage degree: {np.mean(degrees):.2f}")
    print(f"Min degree: {np.min(degrees)}")
    print(f"Max degree: {np.max(degrees)}")


class DoubleConv(nn.Module):
    """(畳み込み => バッチ正規化 => ReLU) を2回繰り返すブロック"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """順伝播"""
        return self.double_conv(x)


class Down(nn.Module):
    """ダウンサンプリングブロック (マックスプーリング + DoubleConv)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """順伝播"""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """アップサンプリングブロック (転置畳み込み + DoubleConv)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 転置畳み込みで空間的な次元を2倍にする
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        # スキップ接続からの特徴マップと結合した後の畳み込み層
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        順伝播
        Args:
            x1 (torch.Tensor): アップサンプリングされる特徴マップ (デコーダからの入力)
            x2 (torch.Tensor): スキップ接続で渡される特徴マップ (エンコーダからの入力)
        """
        # x1をアップサンプリング
        x1 = self.up(x1)
        # エンコーダからの特徴マップ(x2)とサイズを合わせるための差分を計算
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # パディングでサイズを調整
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # チャネル次元でスキップ接続の特徴マップと結合
        x = torch.cat([x2, x1], dim=1)
        # 畳み込み処理
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
        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_rate
        )
        # 3層目: 出力層。ヘッドのアベレージングを行うため concat=False に設定
        self.conv3 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout_rate
        )

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

        # 1D Conv Encoder (Waveform -> Latent Feature)
        self.encoder_dim_1d_conv = (
            512  # Latent feature dimension (analogous to frequency bins)
        )
        self.sampling_rate = 16000
        self.win = 4  # ms (window length)
        self.win_samples = int(self.sampling_rate * self.win / 1000)
        self.stride_samples = self.win_samples // 2

        self.initial_encoder = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=self.encoder_dim_1d_conv,
            kernel_size=self.win_samples,
            bias=False,
            stride=self.stride_samples,
        )

        # GNN Encoder (Operating on the latent features)
        # GNNの入出力次元は、initial_encoderの出力次元とU-Netの入力次元に合わせる
        if gnn_type == "GCN":
            self.gnn_encoder = GCN(
                self.encoder_dim_1d_conv, hidden_dim_gnn, self.encoder_dim_1d_conv
            )
        elif gnn_type == "GAT":
            self.gnn_encoder = GAT(
                self.encoder_dim_1d_conv,
                hidden_dim_gnn,
                self.encoder_dim_1d_conv,
                heads=gnn_heads,
                dropout_rate=gnn_dropout,
            )
        else:
            raise ValueError(
                f"Unsupported GNN type: {gnn_type}. Choose 'GCN' or 'GAT'."
            )

        # U-Net Encoder path
        # GNNの出力は [B, C_feat, L_encoded] となり、これをunsqueezeしてU-Netの2DConvの入力とする
        # したがって、incのin_channelsは1となる
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)  # Bottleneck channel size for U-Net

        # U-Net Decoder path
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        # Output convolution for the mask
        self.outc = nn.Sequential(
            nn.Conv2d(64, self.n_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),  # Mask values between 0 and 1
        )

        # Final 1D ConvTranspose Decoder (Masked latent feature -> Waveform)
        self.final_decoder = nn.ConvTranspose1d(
            in_channels=self.encoder_dim_1d_conv,
            out_channels=self.n_channels,
            kernel_size=self.win_samples,
            bias=False,
            stride=self.stride_samples,
        )

    def create_knn_graph(self, x_nodes_batched, k, batch_size, num_nodes_per_sample):
        """
        バッチ内の各サンプルに対してk-NNグラフを作成します。
        Args:
            x_nodes_batched (torch.Tensor): ノード特徴量 [batch_size * num_nodes_per_sample, num_features]
            k (int): 接続する最近傍ノードの数
            batch_size (int): バッチサイズ
            num_nodes_per_sample (int): 1サンプルあたりのノード数
        Returns:
            torch.Tensor: エッジインデックス [2, num_edges]
        """
        batch_indices = torch.arange(
            batch_size, device=x_nodes_batched.device
        ).repeat_interleave(num_nodes_per_sample)
        edge_index = knn_graph(
            x=x_nodes_batched, k=k, batch=batch_indices, loop=False
        )  # 自己ループなし
        return edge_index

    def forward(self, x_waveform):
        """
        順伝播
        Args:
            x_waveform (torch.Tensor): 入力音声波形 [バッチ, チャネル数, 時間長] または [バッチ, 時間長]
        """
        # Ensure input waveform is 3D: [B, C, L]
        if x_waveform.dim() == 2:
            x_waveform = x_waveform.unsqueeze(1)  # Add channel dimension if missing

        # Store original waveform length for trimming the final output
        original_waveform_length = x_waveform.size(-1)

        # 1. Initial 1D Convolutional Encoder (Waveform -> Latent Feature)
        # x_encoded: [B, encoder_dim_1d_conv, L_encoded]
        x_encoded = self.initial_encoder(x_waveform)

        batch_size, feature_dim_encoded, length_encoded = x_encoded.size()

        # Reshape for GNN: [B, C, L_encoded] -> [B * L_encoded, C]
        # 各時間フレームがノードとなり、その特徴量はencoder_dim_1d_conv次元
        x_nodes = x_encoded.permute(0, 2, 1).reshape(-1, feature_dim_encoded)

        # Create graph for GNN
        num_nodes_per_sample = length_encoded
        if num_nodes_per_sample > 0:
            edge_index = self.create_knn_graph(
                x_nodes, self.num_node_gnn, batch_size, num_nodes_per_sample
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x_nodes.device)

        # グラフの可視化（デバッグ時のみ実行）
        visualize_graph_connections(x_nodes, edge_index)

        # 2. GNN Encoder (Process nodes using GNN)
        x_gnn_out_flat = self.gnn_encoder(
            x_nodes, edge_index
        )  # [B * L_encoded, feature_dim_encoded]

        # Reshape GNN output back to feature map format for U-Net
        # [B * L_encoded, C] -> [B, L_encoded, C] -> [B, C, L_encoded]
        x_gnn_out_reshaped = x_gnn_out_flat.view(
            batch_size, length_encoded, feature_dim_encoded
        ).permute(0, 2, 1)

        # Input to U-Net: [B, 1, C, L_encoded] (C is feature_dim_encoded)
        x_unet_input = x_gnn_out_reshaped.unsqueeze(1)

        # 3. U-Net Encoder path
        x1 = self.inc(x_unet_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # Bottleneck feature map

        # 4. U-Net Decoder path
        d3 = self.up1(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        # 5. Generate Mask
        mask_pred_raw = self.outc(d1)  # [B, n_classes, H_mask, W_mask]

        # Resize mask to match the dimensions of x_gnn_out_reshaped (feature_dim_encoded, length_encoded)
        mask_target_H = feature_dim_encoded
        mask_target_W = length_encoded

        if (
            mask_pred_raw.size(2) != mask_target_H
            or mask_pred_raw.size(3) != mask_target_W
        ):
            mask_resized = F.interpolate(
                mask_pred_raw,
                size=(mask_target_H, mask_target_W),
                mode="bilinear",
                align_corners=False,
            )
        else:
            mask_resized = mask_pred_raw

        # Apply mask to the output of the GNN encoder (x_gnn_out_reshaped)
        # x_gnn_out_reshaped: [B, C, L_encoded]
        # mask_resized: [B, n_classes, C, L_encoded]
        if self.n_classes == 1:
            masked_features = x_gnn_out_reshaped * mask_resized.squeeze(
                1
            )  # Broadcast mask across features
        else:
            # If multiple classes are intended, this needs more specific logic.
            # For typical audio enhancement, n_classes=1 (single mask).
            print(
                f"Warning: GNNEncoder - n_classes > 1 ({self.n_classes}). Using first mask channel."
            )
            masked_features = x_gnn_out_reshaped * mask_resized[:, 0, :, :]

        # 6. Final 1D Convolutional Decoder (Masked latent feature -> Waveform)
        output_waveform = self.final_decoder(masked_features)

        # Trim output waveform to match original input length
        # This handles potential padding introduced by the initial encoder/decoder stride.
        output_waveform = output_waveform[:, :, :original_waveform_length]

        return output_waveform


def print_model_summary(model, batch_size, channels, length):
    """モデルのサマリーを表示するヘルパー関数"""
    x = torch.randn(batch_size, channels, length).to(device)
    print(f"\n{model.__class__.__name__} Model Summary:")
    summary(model, input_data=x)


if __name__ == "__main__":
    print("GNN_encoder.py main execution for model testing")

    # モデルのパラメータ設定
    batch = 1  # バッチサイズ
    num_mic = 1  # マイクの数 (入力チャンネル数)
    length = 16000 * 3  # 音声の長さ (サンプル数): 3秒

    # GNNEncoderUNetモデルのインスタンス化 (GCNタイプ)
    print("\n--- GNNEncoder (GCN as Encoder) ---")
    gnn_encoder_unet_gcn = GNNEncoder(
        n_channels=num_mic, n_classes=1, hidden_dim_gnn=32, num_node=8, gnn_type="GCN"
    ).to(device)
    print_model_summary(gnn_encoder_unet_gcn, batch, num_mic, length)

    # GNNEncoderUNetモデルのインスタンス化 (GATタイプ)
    print("\n--- GNNEncoder (GAT as Encoder) ---")
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

    print("\n--- Forward pass example (GCNEncoderUNet) ---")
    with torch.no_grad():
        output_gcn = gnn_encoder_unet_gcn(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_gcn.shape}")

    print("\n--- Forward pass example (GATEncoderUNet) ---")
    with torch.no_grad():
        output_gat = gnn_encoder_unet_gat(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_gat.shape}")

    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage (after initializations and forward passes):")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
