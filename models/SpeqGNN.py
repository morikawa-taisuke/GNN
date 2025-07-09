import os

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import knn_graph
from torchinfo import summary

# PyTorchのCUDAメモリ管理設定。セグメントを拡張可能にすることで、断片化によるメモリ不足エラーを緩和します。
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAが利用可能かチェックし、利用可能ならGPUを、そうでなければCPUを使用するデバイスとして設定します。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"SpeqGNN.py using device: {device}")


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


class SpeqGCNNet(nn.Module):
    """
    U-NetアーキテクチャのボトルネックにGCNを統合した音声強調/分離モデル。
    このバージョンでは、ランダムなスパースグラフを生成します。
    """
    def __init__(self, n_channels, n_classes, hidden_dim=32, num_node=8, n_fft=512, hop_length=256, win_length=None):
        """
        Args:
            n_channels (int): 時間周波数領域の特徴マップの入力チャネル数 (例: マグニチュードスペクトログラムなら1)
            n_classes (int): 出力マスクのチャネル数 (通常は1)
            hidden_dim (int): GCNの隠れ層の次元数
            num_node (int): k-NNグラフを作成する際の近傍ノード数
            n_fft (int): STFT/ISTFTのFFTサイズ
            hop_length (int): STFT/ISTFTのホップ長
            win_length (int): STFT/ISTFTの窓長 (Noneの場合はn_fftと同じ)
        """
        super(SpeqGCNNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_node = num_node

        # ISTFT（逆短時間フーリエ変換）用のパラメータ
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        # ISTFTで使用する窓関数テンソル。forwardパスで適切なデバイスに移動されます。
        self.window = torch.hann_window(self.win_length)

        # U-Net エンコーダ部分
        # 入力 `x` は既に時間周波数表現 [B, C_feat, F, T] となっています。
        # そのため、self.inc は n_channels (C_feat) を入力として受け取ります。
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)  # ボトルネックのチャネル数は512

        # ボトルネック部分のGNN
        self.gnn = GCN(512, hidden_dim, 512)  # GNNの入出力次元はボトルネックのチャネル数と一致

        # U-Net デコーダ部分
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)  # マスク生成用の出力層

    def create_graph(self, num_nodes):
        """
        ノードごとにランダムにk個の異なる隣接ノードを選択してスパースグラフを作成します。
        自己ループ（自分自身へのエッジ）は作成されません。
        この実装では、torch.topkを使用して処理をベクトル化し、Pythonループを削減しています。
        注意: num_nodesが大きい場合、num_nodes x num_nodes の行列を一時的に使用するため、
              メモリ使用量が増加する可能性があります。
        """
        if num_nodes == 0 or self.num_node == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        # 実際に選択する隣接ノードの数 (自分以外のノード数を超えないように調整)
        if num_nodes == 1:  # ノードが1つの場合、隣接ノードは存在しない
            k_to_select = 0
        else:
            k_to_select = min(self.num_node, num_nodes - 1)

        if k_to_select == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        # 全ての可能なエッジに対してランダムなスコアを生成 (num_nodes x num_nodes 行列)
        # この部分がメモリを大量に消費する可能性があります。
        rand_values = torch.rand(num_nodes, num_nodes, device=device)

        # 自己ループを防ぐため、対角成分を低い値(-1.0)に設定します。
        # torch.rand は [0, 1) の範囲の値を生成するため、-1.0 は選択されません。
        rand_values.fill_diagonal_(-1.0)

        # 各ノードに対して、スコア上位 k_to_select 個のノードのインデックスを取得します。
        top_k_indices = torch.topk(rand_values, k_to_select, dim=1).indices

        # ソースノードのリストを作成: [0,0,...,0, 1,1,...,1, ..., N-1,...,N-1]
        # 各ノードiがk_to_select回繰り返されます。
        source_nodes = torch.arange(num_nodes, device=device).repeat_interleave(k_to_select)

        # ターゲットノードのリストを作成 (top_k_indicesをフラット化)
        target_nodes = top_k_indices.flatten()

        # エッジインデックスを作成 [2, num_edges]
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        return edge_index

    def forward(self, x_magnitude, complex_spec_input, original_length=None, edge_index=None):
        """
        順伝播
        Args:
            x_magnitude (torch.Tensor): 入力マグニチュードスペクトログラム [バッチ, チャネル数, 周波数ビン, 時間フレーム]
            complex_spec_input (torch.Tensor): 元の混合信号の複素スペクトログラム [バッチ, 周波数ビン, 時間フレーム]
            original_length (int, optional): 元の波形データの長さ。ISTFTのlength引数に使用されます。
        """
        input_freq_bins = x_magnitude.size(2)
        input_time_frames = x_magnitude.size(3)

        # 1. U-Net エンコーダ
        x1 = self.inc(x_magnitude)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # ボトルネック特徴マップ: [B, 512, F_bottle, T_bottle]

        # 2. ボトルネックでのGNN処理
        batch_size, channels_bottleneck, height_bottleneck, width_bottleneck = x4.size()
        # GNNに入力するためにテンソルの形状を変更: [B, C, H, W] -> [B, H*W, C] -> [B*H*W, C]
        x4_reshaped = x4.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)

        if edge_index is None:
            # スパースグラフを作成します。
            # 注意: 現在の実装は、バッチ内の全ノードを1つの大きなグラフとして扱います。
            # バッチ内のサンプルごとにグラフを独立させたい場合は、実装の変更が必要です。
            num_nodes = x4_reshaped.size(0)
            edge_index = self.create_graph(num_nodes)

        # GNNによるノード特徴の更新
        x4_processed_flat = self.gnn(x4_reshaped, edge_index)

        # U-Netのデコーダに戻すために形状を復元: [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        x4_processed = x4_processed_flat.view(batch_size, height_bottleneck, width_bottleneck,
                                              channels_bottleneck).permute(0, 3, 1, 2)

        # 3. U-Net デコーダ
        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        # 4. マスクの予測
        mask_pred_raw = self.outc(d1)  # [B, n_classes, F_mask, T_mask]
        mask_pred = torch.sigmoid(mask_pred_raw) # シグモイド関数でマスクを[0, 1]の範囲に正規化

        # 5. マスクを入力特徴マップのサイズにリサイズ
        if mask_pred.size(2) != input_freq_bins or mask_pred.size(3) != input_time_frames:
            mask_pred_resized = F.interpolate(mask_pred, size=(input_freq_bins, input_time_frames), mode='bilinear',
                                              align_corners=False)
        else:
            mask_pred_resized = mask_pred

        # 6. マスクの適用
        # x_magnitude: [B, n_channels, F, T]
        # mask_pred_resized: [B, n_classes, F, T]
        # n_classesが1の場合、マスクは入力チャネル全体にブロードキャストされます。
        predicted_magnitude_tf = x_magnitude * mask_pred_resized  # [B, n_channels(or n_classes), F, T]

        # 7. ISTFTによる波形再構成
        # n_classes=1を想定。predicted_magnitude_tfは [B, 1, F, T]
        # complex_spec_inputは [B, F, T]
        if predicted_magnitude_tf.size(1) == 1:
            predicted_magnitude_for_istft = predicted_magnitude_tf.squeeze(1)  # [B, F, T]
        else:
            # n_classes > 1 の場合は特定の処理が必要。ここでは最初のチャネルを目的のマグニチュードと仮定します。
            print(
                f"Warning: SpeqGCNNet.forward - n_classes > 1 ({predicted_magnitude_tf.size(1)}), ISTFTには最初のチャネルを使用します。")
            predicted_magnitude_for_istft = predicted_magnitude_tf[:, 0, :, :]

        # 元の複素スペクトログラムから位相情報を取得
        phase = torch.angle(complex_spec_input)  # [B, F, T]

        # 予測したマグニチュードと元の位相から複素スペクトログラムを再構成
        reconstructed_complex_spec = torch.polar(predicted_magnitude_for_istft, phase)  # [B, F, T]

        # ISTFTで時間領域の波形に戻す
        output_waveform = torch.istft(reconstructed_complex_spec,
                                      n_fft=self.n_fft,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length,
                                      window=self.window.to(reconstructed_complex_spec.device),
                                      return_complex=False,
                                      length=original_length)  # [B, L_output]
        return output_waveform


class SpeqGATNet(SpeqGCNNet):
    """ SpeqGCNNetのGNN部分をGATに置き換えたモデル """
    def __init__(self, n_channels, n_classes, hidden_dim=32, num_node=8, gat_heads=8, gat_dropout=0.5,
                 n_fft=512, hop_length=256, win_length=None):
        super(SpeqGATNet, self).__init__(n_channels, n_classes, hidden_dim, num_node, n_fft=n_fft,
                                         hop_length=hop_length, win_length=win_length)
        # GNN部分をGATで上書き
        self.gnn = GAT(512, hidden_dim, 512, heads=gat_heads, dropout_rate=gat_dropout)
    # forward と create_graph メソッドは SpeqGCNNet から継承されます


class SpeqGCNNet2(SpeqGCNNet):
    """ SpeqGCNNetのグラフ作成方法を、特徴量ベースのk-NNに変更したバージョン """
    def create_graph(self, x_nodes_batched, k, batch_size, num_nodes_per_sample):
        """
        与えられたノード特徴量に対してk-NNグラフを作成します。
        Args:
            x_nodes_batched (torch.Tensor): ノード特徴量 [batch_size * num_nodes_per_sample, num_features]
            k (int): 接続する最近傍ノードの数
            batch_size (int): バッチサイズ
            num_nodes_per_sample (int): 1サンプルあたりのノード数
        Returns:
            torch.Tensor: エッジインデックス [2, num_edges]
        """
        # バッチ内の各サンプルが独立したグラフになるように、バッチインデックスを作成
        batch_indices = torch.arange(batch_size, device=x_nodes_batched.device).repeat_interleave(num_nodes_per_sample)
        # torch_geometricのknn_graphを使用して、バッチ処理に対応したk-NNグラフを効率的に作成
        edge_index = knn_graph(x=x_nodes_batched, k=k, batch=batch_indices, loop=False)  # 自己ループなし
        return edge_index

    def forward(self, x_magnitude, complex_spec_input, original_length=None, edge_index=None):
        """
        順伝播 (k-NNグラフを使用)
        Args:
            x_magnitude (torch.Tensor): 入力マグニチュードスペクトログラム [B, C, F, T]
            complex_spec_input (torch.Tensor): 元の混合信号の複素スペクトログラム [B, F, T]
            original_length (int, optional): 元の波形データの長さ
        """
        input_freq_bins = x_magnitude.size(2)
        input_time_frames = x_magnitude.size(3)

        # 1. U-Net エンコーダ
        x1 = self.inc(x_magnitude)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # ボトルネック特徴マップ: [B, 512, F_bottle, T_bottle]

        # 2. ボトルネックでのGNN処理
        batch_size, channels_bottleneck, height_bottleneck, width_bottleneck = x4.size()
        # GNNに入力するためにテンソルの形状を変更
        x4_reshaped = x4.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)

        # 特徴量に基づいてk-NNグラフを作成
        num_nodes_per_sample = height_bottleneck * width_bottleneck
        if num_nodes_per_sample > 0:
            edge_index = self.create_graph(x4_reshaped, self.num_node, batch_size, num_nodes_per_sample)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x4_reshaped.device)

        # GNNによるノード特徴の更新
        x4_processed_flat = self.gnn(x4_reshaped, edge_index)

        # U-Netのデコーダに戻すために形状を復元
        x4_processed = x4_processed_flat.view(batch_size, height_bottleneck, width_bottleneck,
                                              channels_bottleneck).permute(0, 3, 1, 2)

        # 3. U-Net デコーダ
        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        # 4. マスクの予測
        mask_pred_raw = self.outc(d1)
        mask_pred = torch.sigmoid(mask_pred_raw)

        # 5. マスクのリサイズ
        if mask_pred.size(2) != input_freq_bins or mask_pred.size(3) != input_time_frames:
            mask_pred_resized = F.interpolate(mask_pred, size=(input_freq_bins, input_time_frames), mode='bilinear',
                                              align_corners=False)
        else:
            mask_pred_resized = mask_pred

        # 6. マスクの適用
        predicted_magnitude_tf = x_magnitude * mask_pred_resized

        # 7. ISTFTによる波形再構成
        if predicted_magnitude_tf.size(1) == 1:
            predicted_magnitude_for_istft = predicted_magnitude_tf.squeeze(1)
        else:
            print(
                f"Warning: SpeqGCNNet.forward - n_classes > 1 ({predicted_magnitude_tf.size(1)}), ISTFTには最初のチャネルを使用します。")
            predicted_magnitude_for_istft = predicted_magnitude_tf[:, 0, :, :]

        phase = torch.angle(complex_spec_input)
        reconstructed_complex_spec = torch.polar(predicted_magnitude_for_istft, phase)

        output_waveform = torch.istft(reconstructed_complex_spec,
                                      n_fft=self.n_fft,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length,
                                      window=self.window.to(reconstructed_complex_spec.device),
                                      return_complex=False,
                                      length=original_length)
        return output_waveform


class SpeqGATNet2(SpeqGCNNet2):
    """ SpeqGCNNet2のGNN部分をGATに置き換えたモデル """
    def __init__(self, n_channels, n_classes, hidden_dim=32, num_node=8, gat_heads=8, gat_dropout=0.5, n_fft=512,
                 hop_length=256, win_length=None):
        super(SpeqGATNet2, self).__init__(n_channels, n_classes, hidden_dim, num_node, n_fft=n_fft,
                                          hop_length=hop_length, win_length=win_length)
        # GNN部分をGATで上書き
        self.gnn = GAT(512, hidden_dim, 512, heads=gat_heads, dropout_rate=gat_dropout)


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

def padding_tensor(tensor1, tensor2):
    """
    最後の次元（例: 時系列長）が異なる2つのテンソルに対して、
    短い方を末尾にゼロパディングして長さをそろえる。

    Args:
        tensor1, tensor2 (torch.Tensor): 任意の次元数のテンソル

    Returns:
        padded_tensor1, padded_tensor2 (torch.Tensor)
    """
    len1 = tensor1.size(-1)
    len2 = tensor2.size(-1)
    max_len = max(len1, len2)

    pad1 = [0, max_len - len1]  # 最後の次元だけパディング
    pad2 = [0, max_len - len2]

    padded_tensor1 = F.pad(tensor1, pad1)
    padded_tensor2 = F.pad(tensor2, pad2)

    return padded_tensor1, padded_tensor2


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

    print("\n--- SpeqGCNNet (Random Graph) ---")
    speq_gcn_model = SpeqGCNNet(n_channels=num_mic, n_classes=1, num_node=8, n_fft=n_fft, hop_length=hop_length, win_length=win_length).to(device)
    print_model_summary(speq_gcn_model, batch, num_mic, length)
    output_gcn = speq_gcn_model(x_magnitude_spec, x_complex_spec, original_len)
    print(f"SpeqGCNNet Input shape: {x_magnitude_spec.shape}, Output shape: {output_gcn.shape}")

    print("\n--- SpeqGATNet (Random Graph, GAT in bottleneck) ---")
    speq_gat_model = SpeqGATNet(n_channels=num_mic, n_classes=1, num_node=8, gat_heads=4, gat_dropout=0.6, n_fft=n_fft, hop_length=hop_length, win_length=win_length).to(device)
    print_model_summary(speq_gat_model, batch, num_mic, length)
    output_gat = speq_gat_model(x_magnitude_spec, x_complex_spec, original_len)
    print(f"SpeqGATNet Input shape: {x_magnitude_spec.shape}, Output shape: {output_gat.shape}")

    print("\n--- SpeqGCNNet2 (k-NN Graph, GCN in bottleneck) ---")
    speq_gcn2_model = SpeqGCNNet2(n_channels=num_mic, n_classes=1, num_node=8, n_fft=n_fft, hop_length=hop_length, win_length=win_length).to(device)
    print_model_summary(speq_gcn2_model, batch, num_mic, length)
    output_gcn2 = speq_gcn2_model(x_magnitude_spec, x_complex_spec, original_len)
    print(f"SpeqGCNNet2 Input shape: {x_magnitude_spec.shape}, Output shape: {output_gcn2.shape}")

    print("\n--- SpeqGATNet2 (k-NN Graph, GAT in bottleneck) ---")
    speq_gat2_model = SpeqGATNet2(n_channels=num_mic, n_classes=1, num_node=8, gat_heads=4, gat_dropout=0.6, n_fft=n_fft, hop_length=hop_length, win_length=win_length).to(device)
    print_model_summary(speq_gat2_model, batch, num_mic, length)
    output_gat2 = speq_gat2_model(x_magnitude_spec, x_complex_spec, original_len)
    print(f"SpeqGATNet2 Input shape: {x_magnitude_spec.shape}, Output shape: {output_gat2.shape}")


    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage (after initializations):")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == '__main__':
    main()
