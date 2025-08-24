import os

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import knn_graph
from torchinfo import summary  # モデルのサマリー表示用

from mymodule import confirmation_GPU

# PyTorchのCUDAメモリ管理設定。セグメントを拡張可能にすることで、断片化によるメモリ不足エラーを緩和します。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CUDAが利用可能かチェックし、利用可能ならGPUを、そうでなければCPUを使用するデバイスとして設定します。
device = confirmation_GPU.get_device()
print(f"SpeqGNN_encoder.py 使用デバイス: {device}")


# --- 既存のユーティリティークラス (SpeqGNN.pyから再利用) ---
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
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
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
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_rate
        )
        self.conv3 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout_rate
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index)
        return x


# --- 新しいモデル: SpeqGCN_encoder ---
class SpeqGCN_encoder(nn.Module):
    """
    スペクトログラム入力に対し、GNN、独立したDoubleConv、その後にU-Netを順に適用するモデル。
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        hidden_dim_gnn=32,
        num_node=8,
        gnn_type="GCN",
        gnn_heads=4,
        gnn_dropout=0.6,
        n_fft=512,
        hop_length=256,
        win_length=None,
    ):
        """
        Args:
            n_channels (int): 入力スペクトログラムのチャネル数 (例: マグニチュードなら1)。
            n_classes (int): 出力マスクのチャネル数 (通常は1)。
            hidden_dim_gnn (int): GNNの隠れ層の次元数。
            num_node (int): k-NNグラフを作成する際の近傍ノード数。
            gnn_type (str): 使用するGNNの種類 ('GCN' または 'GAT')。
            gnn_heads (int): GATの場合のマルチヘッドアテンションのヘッド数。
            gnn_dropout (float): GATの場合のドロップアウト率。
            n_fft (int): ISTFTのFFTサイズ。
            hop_length (int): ISTFTのホップ長。
            win_length (int): ISTFTの窓長 (Noneの場合はn_fftと同じ)。
        """
        super(SpeqGCN_encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_node_gnn = num_node

        # ISTFT用のパラメータと窓関数
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = torch.hann_window(self.win_length)

        # GNNの出力チャネル数を設定 (U-Netの inc 層の入力チャネルに合わせる)
        gnn_output_channels = 64

        # 1. GNN層 (入力スペクトログラムを直接処理)
        # GNNの入力次元は、スペクトログラムのチャネル数 (各ノードの特徴量次元)
        if gnn_type == "GCN":
            self.initial_gnn = GCN(n_channels, hidden_dim_gnn, gnn_output_channels)
        elif gnn_type == "GAT":
            self.initial_gnn = GAT(
                n_channels,
                hidden_dim_gnn,
                gnn_output_channels,
                heads=gnn_heads,
                dropout_rate=gnn_dropout,
            )
        else:
            raise ValueError(
                f"Unsupported GNN type: {gnn_type}. Choose 'GCN' or 'GAT'."
            )

        # 2. GNNの出力に続くDoubleConv層
        # GNNの出力チャネルがこのDoubleConvの入力となり、U-Netのinc層の入力チャネルに合わせる
        self.intermediate_double_conv = DoubleConv(gnn_output_channels, 64)

        # 3. フルU-Net構造 (独自のエンコーダーとデコーダーを持つ)
        # U-Netの inc 層は、中間DoubleConvの出力チャネル (64) を入力とする
        self.unet_inc = DoubleConv(64, 64)
        self.unet_down1 = Down(64, 128)
        self.unet_down2 = Down(128, 256)
        self.unet_down3 = Down(256, 512)  # U-Net内部のボトルネックチャネル

        self.unet_up1 = Up(512, 256)
        self.unet_up2 = Up(256, 128)
        self.unet_up3 = Up(128, 64)

        # U-Netの出力層 (マスク生成)
        self.unet_outc = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),  # マスクを [0, 1] に正規化
        )

    def create_knn_graph(self, x_nodes_batched, k, batch_size, num_nodes_per_sample):
        """
        与えられたノード特徴量に対してk-NNグラフを作成します (SpeqGCNNet2から再利用)。
        """
        batch_indices = torch.arange(
            batch_size, device=x_nodes_batched.device
        ).repeat_interleave(num_nodes_per_sample)
        edge_index = knn_graph(x=x_nodes_batched, k=k, batch=batch_indices, loop=False)
        return edge_index

    def forward(self, x_magnitude, complex_spec_input, original_length=None):
        """
        順伝播
        Args:
            x_magnitude (torch.Tensor): 入力マグニチュードスペクトログラム [B, C, F, T]
            complex_spec_input (torch.Tensor): 元の混合信号の複素スペクトログラム [B, F, T]
            original_length (int, optional): 元の波形データの長さ。ISTFTのlength引数に使用されます。
        """
        input_freq_bins = x_magnitude.size(2)
        input_time_frames = x_magnitude.size(3)

        # 1. GNN処理 (入力スペクトログラムをノードとして扱う)
        batch_size, num_channels_input, height_input, width_input = x_magnitude.size()
        # GNNに入力するためにテンソルをノードの形に整形: [B, C, F, T] -> [B*F*T, C]
        x_nodes_for_gnn = (
            x_magnitude.reshape(batch_size, num_channels_input, -1)
            .permute(0, 2, 1)
            .reshape(-1, num_channels_input)
        )

        # k-NNグラフの作成
        num_nodes_per_sample = height_input * width_input
        if num_nodes_per_sample > 0:
            edge_index = self.create_knn_graph(
                x_nodes_for_gnn, self.num_node_gnn, batch_size, num_nodes_per_sample
            )
        else:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=x_nodes_for_gnn.device
            )

        # GNNによるノード特徴の更新
        gnn_output_flat = self.initial_gnn(x_nodes_for_gnn, edge_index)

        # GNNの出力を2D特徴マップの形に復元: [B*F*T, C_out_gnn] -> [B, C_out_gnn, F, T]
        gnn_output_reshaped = gnn_output_flat.view(
            batch_size, height_input, width_input, gnn_output_flat.size(-1)
        ).permute(0, 3, 1, 2)

        # 2. 中間のDoubleConv層
        intermediate_conv_output = self.intermediate_double_conv(gnn_output_reshaped)

        # 3. フルU-Net構造
        # U-Netエンコーダーパス
        u_x1 = self.unet_inc(
            intermediate_conv_output
        )  # intermediate_conv_output がU-Netの入力
        u_x2 = self.unet_down1(u_x1)
        u_x3 = self.unet_down2(u_x2)
        u_x4 = self.unet_down3(u_x3)  # U-Net内部のボトルネック特徴マップ

        # U-Netデコーダーパス (スキップ接続を使用)
        u_d3 = self.unet_up1(u_x4, u_x3)
        u_d2 = self.unet_up2(u_d3, u_x2)
        u_d1 = self.unet_up3(u_d2, u_x1)

        # 4. マスクの予測
        mask_pred = self.unet_outc(u_d1)  # マスクは[0, 1]に正規化済み

        # 5. マスクを入力スペクトログラムのサイズにリサイズ
        if (
            mask_pred.size(2) != input_freq_bins
            or mask_pred.size(3) != input_time_frames
        ):
            mask_pred_resized = F.interpolate(
                mask_pred,
                size=(input_freq_bins, input_time_frames),
                mode="bilinear",
                align_corners=False,
            )
        else:
            mask_pred_resized = mask_pred

        # 6. マスクの適用
        predicted_magnitude_tf = x_magnitude * mask_pred_resized

        # 7. ISTFTによる波形再構成
        if predicted_magnitude_tf.size(1) == 1:
            predicted_magnitude_for_istft = predicted_magnitude_tf.squeeze(1)
        else:
            print(
                f"Warning: SpeqGCN_encoder.forward - n_classes > 1 ({predicted_magnitude_tf.size(1)}), ISTFTには最初のチャネルを使用します。"
            )
            predicted_magnitude_for_istft = predicted_magnitude_tf[:, 0, :, :]

        phase = torch.angle(complex_spec_input)
        reconstructed_complex_spec = torch.polar(predicted_magnitude_for_istft, phase)

        output_waveform = torch.istft(
            reconstructed_complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(reconstructed_complex_spec.device),
            return_complex=False,
            length=original_length,
        )
        return output_waveform


class SpeqGAT_encoder(SpeqGCN_encoder):
    """SpeqGNNPreUNetのGNN部分をGATに置き換えたモデル"""

    def __init__(
        self,
        n_channels,
        n_classes,
        hidden_dim_gnn=32,
        num_node=8,
        gnn_heads=8,
        gnn_dropout=0.5,
        n_fft=512,
        hop_length=256,
        win_length=None,
    ):
        super(SpeqGAT_encoder, self).__init__(
            n_channels,
            n_classes,
            hidden_dim_gnn,
            num_node,
            gnn_type="GAT",
            gnn_heads=gnn_heads,
            gnn_dropout=gnn_dropout,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        # GNN部分をGATで上書き (SpeqGNNPreUNetのコンストラクタで既に設定されているが、明示的にGATとしてインスタンス化)
        gnn_input_dim = n_channels
        gnn_output_channels = 64
        self.initial_gnn = GAT(
            gnn_input_dim,
            hidden_dim_gnn,
            gnn_output_channels,
            heads=gnn_heads,
            dropout_rate=gnn_dropout,
        )


# --- 使用例 ---
if __name__ == "__main__":
    print("SpeqGNN_PreUNet.py main execution for model testing")

    # モデルのパラメータ設定
    batch_size = 1
    n_channels = 1  # マグニチュードスペクトログラムの入力チャネル (モノラル音声)
    n_classes = 1  # マスクの出力チャネル

    # STFT/ISTFTパラメータ
    n_fft_val = 512
    hop_length_val = 256
    win_length_val = 512

    # ダミーのスペクトログラムデータを作成
    # 例: 音声が16kHzで3秒の場合、48000サンプル
    # 時間フレーム数 = floor((48000 - win_length) / hop_length) + 1 = floor((48000 - 512) / 256) + 1 = 186
    # 周波数ビン数 = n_fft / 2 + 1 = 512 / 2 + 1 = 257
    dummy_freq_bins = n_fft_val // 2 + 1
    dummy_time_frames = 188  # 任意のフレーム数
    original_audio_length = 48000  # 元の波形サンプル長

    # マグニチュードスペクトログラム: [B, C, F, T]
    dummy_magnitude_spec = torch.randn(
        batch_size, n_channels, dummy_freq_bins, dummy_time_frames
    ).to(device)
    # 複素スペクトログラム: [B, F, T] (torchaudio.transforms.Spectrogram の return_complex=True の出力形式)
    dummy_complex_spec = torch.randn(
        batch_size, dummy_freq_bins, dummy_time_frames, dtype=torch.complex64
    ).to(device)

    # SpeqGCN_encoder (GCN) のインスタンス化とサマリー表示
    print("\n--- SpeqGCN_encoder (GCN) ---")
    model_gcn = SpeqGCN_encoder(
        n_channels=n_channels,
        n_classes=n_classes,
        hidden_dim_gnn=32,
        num_node=8,
        gnn_type="GCN",
        n_fft=n_fft_val,
        hop_length=hop_length_val,
        win_length=win_length_val,
    ).to(device)
    print("GCN Model Summary:")
    summary(
        model_gcn,
        input_data=(dummy_magnitude_spec, dummy_complex_spec, original_audio_length),
    )

    # SpeqGAT_encoder (GAT) のインスタンス化とサマリー表示
    print("\n--- SpeqGAT_encoder (GAT) ---")
    model_gat = SpeqGAT_encoder(
        n_channels=n_channels,
        n_classes=n_classes,
        hidden_dim_gnn=32,
        num_node=8,
        gnn_heads=4,
        gnn_dropout=0.6,
        n_fft=n_fft_val,
        hop_length=hop_length_val,
        win_length=win_length_val,
    ).to(device)
    print("GAT Model Summary:")
    summary(
        model_gat,
        input_data=(dummy_magnitude_spec, dummy_complex_spec, original_audio_length),
    )

    # フォワードパスの実行例 (GCNモデル)
    print("\n--- Forward pass example (SpeqGCN_encoder - GCN) ---")
    with torch.no_grad():
        output_gcn = model_gcn(
            dummy_magnitude_spec, dummy_complex_spec, original_audio_length
        )
    print(f"Input magnitude spec shape: {dummy_magnitude_spec.shape}")
    print(f"Input complex spec shape: {dummy_complex_spec.shape}")
    print(f"Output waveform shape: {output_gcn.shape}")  # [B, L_output]

    # フォワードパスの実行例 (GATモデル)
    print("\n--- Forward pass example (SpeqGAT_encoder - GAT) ---")
    with torch.no_grad():
        output_gat = model_gat(
            dummy_magnitude_spec, dummy_complex_spec, original_audio_length
        )
    print(f"Input magnitude spec shape: {dummy_magnitude_spec.shape}")
    print(f"Input complex spec shape: {dummy_complex_spec.shape}")
    print(f"Output waveform shape: {output_gat.shape}")  # [B, L_output]

    # GPUメモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage (after initializations and forward passes):")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
