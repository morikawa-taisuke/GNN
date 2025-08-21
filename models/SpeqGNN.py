import os

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import knn_graph
from torchinfo import summary

from mymodule import confirmation_GPU

# PyTorchのCUDAメモリ管理設定。セグメントを拡張可能にすることで、断片化によるメモリ不足エラーを緩和します。
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAが利用可能かチェックし、利用可能ならGPUを、そうでなければCPUを使用するデバイスとして設定します。
device = confirmation_GPU.get_device()
print(f"SpeqGNN.py 使用デバイス: {device}")


class DoubleConv(nn.Module):
    """ (畳み込み => バッチ正規化 => ReLU) を2回繰り返すブロック """
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
        """ 順伝播 """
        return self.double_conv(x)


class Down(nn.Module):
    """ ダウンサンプリングブロック (マックスプーリング + DoubleConv) """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """ 順伝播 """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """ アップサンプリングブロック (転置畳み込み + DoubleConv) """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 転置畳み込みで空間的な次元を2倍にする
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # チャネル次元でスキップ接続の特徴マップと結合
        x = torch.cat([x2, x1], dim=1)
        # 畳み込み処理
        return self.conv(x)


class GCN(nn.Module):
    """ 3層のGraph Convolutional Network (GCN) モデル """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """ 順伝播 """
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class GAT(nn.Module):
    """ 3層のGraph Attention Network (GAT) モデル """
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
        """ 順伝播 """
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class SpeqUNetGNN(nn.Module):
    """
    U-NetアーキテクチャのボトルネックにGNNを統合した音声強調/分離モデル。
    GNNの種類とグラフの作成方法をパラメータで指定できる。
    """
    def __init__(self, n_channels, n_classes,
                 gnn_type="GCN",
                 graph_creation="knn",  # "knn" または "random"
                 hidden_dim=32,
                 num_node=8,
                 gat_heads=8,
                 gat_dropout=0.5,
                 n_fft=512,
                 hop_length=256,
                 win_length=None):
        """
        Args:
            n_channels (int): 時間周波数領域の特徴マップの入力チャネル数 (例: マグニチュードスペクトログラムなら1)
            n_classes (int): 出力マスクのチャネル数 (通常は1)
            gnn_type (str): 使用するGNNの種類 ("GCN" or "GAT")
            graph_creation (str): グラフの作成方法 ("knn" or "random")
            hidden_dim (int): GNNの隠れ層の次元数
            num_node (int): k-NNグラフを作成する際の近傍ノード数
            gat_heads (int): GATで使用するヘッド数
            gat_dropout (float): GATのドロップアウト率
            n_fft (int): STFT/ISTFTのFFTサイズ
            hop_length (int): STFT/ISTFTのホップ長
            win_length (int): STFT/ISTFTの窓長 (Noneの場合はn_fftと同じ)
        """
        super(SpeqUNetGNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_node = num_node
        self.graph_creation = graph_creation

        # ISTFT（逆短時間フーリエ変換）用のパラメータ
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = torch.hann_window(self.win_length)

        # U-Net エンコーダ部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # ボトルネック部分のGNN
        if gnn_type.upper() == "GCN":
            self.gnn = GCN(512, hidden_dim, 512)
        elif gnn_type.upper() == "GAT":
            self.gnn = GAT(512, hidden_dim, 512, heads=gat_heads, dropout_rate=gat_dropout)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        # U-Net デコーダ部分
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

    def _create_random_graph(self, num_nodes, device):
        """
        ノードごとにランダムにk個の異なる隣接ノードを選択してスパースグラフを作成します。
        """
        if num_nodes <= 1 or self.num_node == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        k_to_select = min(self.num_node, num_nodes - 1)
        if k_to_select == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        rand_values = torch.rand(num_nodes, num_nodes, device=device)
        rand_values.fill_diagonal_(-1.0)
        top_k_indices = torch.topk(rand_values, k_to_select, dim=1).indices
        source_nodes = torch.arange(num_nodes, device=device).repeat_interleave(k_to_select)
        target_nodes = top_k_indices.flatten()
        return torch.stack([source_nodes, target_nodes], dim=0)

    def _create_knn_graph(self, x_nodes_batched, k, batch_size, num_nodes_per_sample):
        """
        与えられたノード特徴量に対してk-NNグラフを作成します。
        """
        if k == 0 or num_nodes_per_sample == 0:
            k_to_select = 0
        else:
            k_to_select = min(k, num_nodes_per_sample - 1) if num_nodes_per_sample > 1 else 0

        if k_to_select == 0:
            return torch.empty((2, 0), dtype=torch.long, device=x_nodes_batched.device)

        batch_indices = torch.arange(batch_size, device=x_nodes_batched.device).repeat_interleave(num_nodes_per_sample)
        return knn_graph(x=x_nodes_batched, k=k_to_select, batch=batch_indices, loop=False)

    def forward(self, x_magnitude, complex_spec_input, original_length=None):
        """
        順伝播
        Args:
            x_magnitude (torch.Tensor): 入力マグニチュードスペクトログラム [バッチ, チャネル数, 周波数ビン, 時間フレーム]
            complex_spec_input (torch.Tensor): 元の混合信号の複素スペクトログラム [バッチ, 周波数ビン, 時間フレーム]
            original_length (int, optional): 元の波形データの長さ。ISTFTのlength引数に使用されます。
        """
        batch_size, _, input_freq_bins, input_time_frames = x_magnitude.size()

        # 1. U-Net エンコーダ
        x1 = self.inc(x_magnitude)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 2. ボトルネックでのGNN処理
        _, channels_bottleneck, height_bottleneck, width_bottleneck = x4.size()
        x4_reshaped = x4.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)

        # グラフ作成
        if self.graph_creation == "random":
            num_nodes = x4_reshaped.size(0)
            edge_index = self._create_random_graph(num_nodes, x4_reshaped.device)
        elif self.graph_creation == "knn":
            num_nodes_per_sample = height_bottleneck * width_bottleneck
            edge_index = self._create_knn_graph(x4_reshaped, self.num_node, batch_size, num_nodes_per_sample)
        else:
            raise ValueError(f"Unknown graph creation method: {self.graph_creation}")

        # GNNによるノード特徴の更新
        x4_processed_flat = self.gnn(x4_reshaped, edge_index)

        x4_processed = x4_processed_flat.view(batch_size, height_bottleneck, width_bottleneck,
                                              channels_bottleneck).permute(0, 3, 1, 2)

        # 3. U-Net デコーダ
        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        # 4. マスクの予測
        mask_pred_raw = self.outc(d1)
        mask_pred = torch.sigmoid(mask_pred_raw)

        # 5. マスクを入力特徴マップのサイズにリサイズ
        if mask_pred.size(2) != input_freq_bins or mask_pred.size(3) != input_time_frames:
            mask_pred_resized = F.interpolate(mask_pred, size=(input_freq_bins, input_time_frames), mode='bilinear',
                                              align_corners=False)
        else:
            mask_pred_resized = mask_pred

        # 6. マスクの適用
        predicted_magnitude_tf = x_magnitude * mask_pred_resized

        # 7. ISTFTによる波形再構成
        if predicted_magnitude_tf.size(1) == 1:
            predicted_magnitude_for_istft = predicted_magnitude_tf.squeeze(1)  # [B, F, T]
        else:
            # n_classes > 1 の場合は特定の処理が必要。ここでは最初のチャネルを目的のマグニチュードと仮定します。
            print(
                f"Warning: SpeqGCNNet.forward - n_classes > 1 ({predicted_magnitude_tf.size(1)}), ISTFTには最初のチャネルを使用します。")
            predicted_magnitude_for_istft = predicted_magnitude_tf[:, 0, :, :]

        phase = torch.angle(complex_spec_input)
        reconstructed_complex_spec = torch.polar(predicted_magnitude_for_istft, phase)

        # ISTFTで時間領域の波形に戻す
        output_waveform = torch.istft(reconstructed_complex_spec,
                                      n_fft=self.n_fft,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length,
                                      window=self.window.to(reconstructed_complex_spec.device),
                                      return_complex=False,
                                      length=original_length)
        return output_waveform


def print_model_summary(model, batch_size, channels, length):
    # --- STFTパラメータ (モデルの設計に合わせて調整してください) ---
    n_fft = 512
    hop_length = n_fft // 2
    win_length = n_fft
    window = torch.hann_window(win_length, device=device)

    # サンプル入力データの作成
    x_time = torch.randn(batch_size, 1, length, device=device)

    # --- 入力データをSTFTでスペクトログラムに変換 ---
    x_magnitude_spec = torch.stft(x_time.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=False)
    x_magnitude_spec = torch.sqrt(x_magnitude_spec[..., 0]**2 + x_magnitude_spec[..., 1]**2).unsqueeze(1) # (B, 1, F, T_spec)

    # 複素スペクトログラム (B, F, T_spec)
    x_complex_spec = torch.stft(x_time.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

    original_len = x_time.shape[-1]

    # 上記のinputデータの形状を確認
    print(f"Input shape: x_magnitude_spec: {x_magnitude_spec.shape}, x_complex_spec: {x_complex_spec.shape}, original_len: {original_len}")

    # モデルのサマリーを表示
    print(f"\n{model.__class__.__name__} Model Summary:")
    summary(model, input_data=(x_magnitude_spec, x_complex_spec, original_len), device=device)


def main():
    print("SpecGNN.py main execution")
    # サンプルデータの作成（入力サイズを縮小）
    batch = 1
    num_mic = 1
    length = 16000 * 8  # 8秒の音声データ (例)

    # --- STFTパラメータ ---
    n_fft = 1024
    hop_length = n_fft // 2
    win_length = n_fft
    window = torch.hann_window(win_length, device=device)

    # --- サンプル入力データの作成 ---
    x_time = torch.randn(batch, 1, length, device=device)
    # マグニチュードスペクトログラム
    stft_result = torch.stft(x_time.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=False)
    x_magnitude_spec = torch.sqrt(stft_result[..., 0]**2 + stft_result[..., 1]**2).unsqueeze(1) # (B, 1, F, T_spec)
    # 複素スペクトログラム
    x_complex_spec = torch.stft(x_time.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    original_len = x_time.shape[-1]

    # --- モデルのインスタンス化とテスト ---
    common_params = {
        "n_channels": num_mic,
        "n_classes": 1,
        "num_node": 8,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length
    }
    gat_params = {
        "gat_heads": 4,
        "gat_dropout": 0.6
    }

    print("\n--- SpeqGCNNet (Random Graph) ---")
    speq_gcn_model = SpeqUNetGNN(**common_params, gnn_type="GCN", graph_creation="random").to(device)
    print_model_summary(speq_gcn_model, batch, num_mic, length)
    output_gcn = speq_gcn_model(x_magnitude_spec, x_complex_spec, original_len)
    print(f"Output shape: {output_gcn.shape}")

    print("\n--- SpeqGATNet (Random Graph, GAT in bottleneck) ---")
    speq_gat_model = SpeqUNetGNN(**common_params, **gat_params, gnn_type="GAT", graph_creation="random").to(device)
    print_model_summary(speq_gat_model, batch, num_mic, length)
    output_gat = speq_gat_model(x_magnitude_spec, x_complex_spec, original_len)
    print(f"Output shape: {output_gat.shape}")

    print("\n--- SpeqGCNNet2 (k-NN Graph, GCN in bottleneck) ---")
    speq_gcn2_model = SpeqUNetGNN(**common_params, gnn_type="GCN", graph_creation="knn").to(device)
    print_model_summary(speq_gcn2_model, batch, num_mic, length)
    output_gcn2 = speq_gcn2_model(x_magnitude_spec, x_complex_spec, original_len)
    print(f"Output shape: {output_gcn2.shape}")

    print("\n--- SpeqGATNet2 (k-NN Graph, GAT in bottleneck) ---")
    speq_gat2_model = SpeqUNetGNN(**common_params, **gat_params, gnn_type="GAT", graph_creation="knn").to(device)
    print_model_summary(speq_gat2_model, batch, num_mic, length)
    output_gat2 = speq_gat2_model(x_magnitude_spec, x_complex_spec, original_len)
    print(f"Output shape: {output_gat2.shape}")


    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage (after initializations):")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == '__main__':
    main()
