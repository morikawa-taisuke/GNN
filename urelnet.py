# urelnet2.py (修正版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchinfo import summary
import os
# from torch_geometric.utils import knn_graph
from torch_geometric.nn import knn_graph

# CUDAのメモリ管理設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAの可用性をチェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class RelNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RelNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class URelNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_dim=32, k_neighbors=8):
        super(URelNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes  # 雑音抑圧・残響除去では通常1 (マスクのチャネル数)
        self.k_neighbors = k_neighbors

        self.encoder_dim = 512  # エンコーダの出力チャネル数（周波数ビンの数に相当）
        self.sampling_rate = 16000
        self.win = 4  # ms
        self.win_samples = int(self.sampling_rate * self.win / 1000)  # 窓長サンプル数
        self.stride_samples = self.win_samples // 2  # 畳み込み処理におけるフィルタが移動する幅

        # エンコーダ（波形 -> 時間-周波数表現）
        self.encoder = nn.Conv1d(in_channels=n_channels,
                                 out_channels=self.encoder_dim,
                                 kernel_size=self.win_samples,
                                 bias=False,
                                 stride=self.stride_samples)

        # デコーダ（マスクされた時間-周波数表現 -> 波形）
        self.decoder = nn.ConvTranspose1d(in_channels=self.encoder_dim,  # マスク適用後のチャネル数はencoder_dim
                                          out_channels=n_channels,
                                          kernel_size=self.win_samples,
                                          bias=False,
                                          stride=self.stride_samples)

        # U-Net エンコーダー部分
        # n_channels (入力波形) -> encoder_dim (エンコーダ出力)
        # したがって、incの入力はencoder_dimチャネルを持つべきです。
        # x_unet_input は x_encoded.unsqueeze(dim=1) で [B, 1, encoder_dim, L_encoded]
        # になるため、incのin_channelsは1が適切です。
        self.inc = DoubleConv(1, 64)  # U-Netの入力チャネルは、x_encodedをunsqueezeした後の次元
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # ボトルネック部分（RelNet）
        # RelNetのinput_dimは、down3の出力チャネルである512
        self.relnet = RelNet(512, hidden_dim, 512)

        # デコーダー部分
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        # 最終的なU-Netの出力層（マスクを生成）
        # 出力チャネルはn_classes（通常1）、Sigmoidで0-1の範囲に正規化
        self.outc = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()  # マスクを0から1の範囲に正規化
        )

    # PyTorch Geometricのknn_graphを使用してグラフを構築
    def create_knn_graph(self, x, k, batch_size, num_nodes_per_sample):
        # x: [batch_size * num_nodes_per_sample, num_features]
        batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes_per_sample)
        edge_index = knn_graph(x=x, k=k, batch=batch_indices, loop=False)
        return edge_index

    def forward(self, x):
        # 1. 生波形を時間-周波数表現にエンコード
        # x: [batch_size, n_channels, length]
        x_encoded = self.encoder(x)  # [batch_size, encoder_dim, L_encoded]

        # 2. U-Netの入力形式に変換
        # U-NetはConv2dを使うため、[batch_size, channels, H, W]の形式が必要
        # x_encoded: [batch_size, encoder_dim, L_encoded]
        # H=encoder_dim, W=L_encodedと解釈されるように次元を追加
        # [batch_size, 1, encoder_dim, L_encoded]
        x_unet_input = x_encoded.unsqueeze(dim=1)

        # 3. U-Net エンコーダー部分
        x1 = self.inc(x_unet_input)  # [B, 64, H1, W1]
        x2 = self.down1(x1)  # [B, 128, H2, W2]
        x3 = self.down2(x2)  # [B, 256, H3, W3]
        x4 = self.down3(x3)  # [B, 512, H4, W4]

        # 4. ボトルネック（RelNet）
        batch_size, channels, height, width = x4.size()  # H4, W4

        # RelNetの入力形式にリシェイプ: [B*H*W, C]
        # x4: [B, C, H, W] -> [B, H*W, C] -> [B*H*W, C]
        x4_reshaped = x4.view(batch_size, channels, -1).permute(0, 2, 1).reshape(-1, channels)

        # RelNet用のグラフを動的に構築
        num_nodes_per_sample = height * width
        edge_index = self.create_knn_graph(x4_reshaped, self.k_neighbors, batch_size, num_nodes_per_sample)

        # RelNetを適用
        x4_processed = self.relnet(x4_reshaped, edge_index)

        # RelNet処理後の特徴を元のU-Netの形状に戻す
        # [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        x4_processed = x4_processed.view(batch_size, height, width, channels).permute(0, 3, 1, 2)

        # 5. U-Net デコーダー部分
        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        # 6. マスクの生成
        # logits: [batch_size, n_classes, H_x1, W_x1] (x1の空間次元に対応)
        mask = self.outc(d1)

        # 7. マスクの適用（`x_encoded`に）
        # `mask`の空間次元を`x_encoded`の空間次元`L_encoded`に合わせる
        # `mask`: [B, n_classes, H_x1, W_x1]
        # `x_encoded`: [B, encoder_dim, L_encoded]

        # まず、`mask`の空間次元 (H_x1, W_x1) をフラット化し、`L_mask`とする
        # H_x1 * W_x1 が `x_encoded` の `L_encoded` と一致するはずですが、
        # そうでない場合のために補間（interpolate）します。
        mask_flat_spatial_dim = mask.view(batch_size, self.n_classes, -1)  # [B, n_classes, H_x1*W_x1]

        target_length = x_encoded.size(2)  # `x_encoded`のL_encoded

        # マスクをターゲットの長さにリサイズ
        # mode='linear' (1D) または 'bilinear' (2D) が適していますが、
        # ここでは mask_flat_spatial_dim が [B, C, L'] の形なので 'linear' が適切
        mask_upsampled = F.interpolate(mask_flat_spatial_dim, size=target_length, mode='linear', align_corners=False)
        # mask_upsampled: [B, n_classes, L_encoded]

        # `n_classes`が1の場合を前提とする
        if self.n_classes == 1:
            # `mask_upsampled`は`[B, 1, L_encoded]`
            # `x_encoded`は`[B, encoder_dim, L_encoded]`
            # マスクを`x_encoded`の各チャネルにブロードキャストして適用
            masked_x_encoded = x_encoded * mask_upsampled  # ブロードキャストにより`encoder_dim`次元に適用
        else:
            # `n_classes`が1でない場合の処理（例：各チャネルのマスクが異なる意味を持つ場合など）
            # モデルの設計意図に依存しますが、雑音抑圧では通常`n_classes=1`です
            raise ValueError("For noise reduction, n_classes is typically 1 for a single mask.")

        # 8. マスク適用後の時間-周波数表現を波形にデコード
        output_waveform = self.decoder(masked_x_encoded)

        return output_waveform


def print_model_summary(model, batch_size, channels, length):
    x = torch.randn(batch_size, channels, length).to(device)
    print("\nURelNet Model Summary:")
    summary(model, input_data=x)


def main():
    print("main")
    batch = 1
    num_mic = 1  # 1チャンネル入力 (モノラル)
    length = 16000 * 5  # 5秒間の音声データ (例)

    x = torch.randn(batch, num_mic, length).to(device)

    # 雑音抑圧・残響除去の場合、n_classesは通常1（単一のマスクを生成）
    model = URelNet(n_channels=num_mic, n_classes=1, k_neighbors=8).to(device)

    print_model_summary(model, batch, num_mic, length)

    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == '__main__':
    main()