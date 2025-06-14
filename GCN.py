import torch
from torch import nn
import os
from torchinfo import summary
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import knn_graph


# CUDAのメモリ管理設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAの可用性をチェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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


class GCN(nn.Module):
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


class UGCNNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_dim=32, k_neighbors=8):
        super(UGCNNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.k_neighbors = k_neighbors

        self.encoder_dim = 512
        self.sampling_rate = 16000
        self.win = 4
        self.win = int(self.sampling_rate * self.win / 1000)
        self.stride = self.win // 2  # 畳み込み処理におけるフィルタが移動する幅

        # エンコーダ
        self.encoder = nn.Conv1d(in_channels=n_channels,  # 入力データの次元数 #=1もともとのやつ
                                 out_channels=self.encoder_dim,  # 出力データの次元数
                                 kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長なの?)
                                 bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
                                 stride=self.stride)  # 畳み込み処理の移動幅
        # デコーダ
        self.decoder = nn.ConvTranspose1d(in_channels=self.encoder_dim,  # 入力次元数
                                          out_channels=n_channels,  # 出力次元数 1もともとのやつ
                                          kernel_size=self.win,  # カーネルサイズ
                                          bias=False,
                                          stride=self.stride)  # 畳み込み処理の移動幅

        # エンコーダー部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # ボトルネック部分（RelNet）
        self.relnet = GCN(512, hidden_dim, 512)

        # デコーダー部分
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

    def create_sparse_graph(self, num_nodes):
        """
        ノードごとにランダムにk個の異なる隣接ノードを選択してスパースグラフを作成します。
        自己ループは作成されません。
        この最適化版では、torch.topkを使用して処理をベクトル化し、Pythonループを削減します。
        注意: num_nodesが大きい場合、num_nodes x num_nodes の行列を一時的に使用するため、
              メモリ使用量が増加する可能性があります。
        """
        if num_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        if self.k_neighbors == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        # 実際に選択する隣接ノードの数 (num_nodes - 1 を超えることはない)
        if num_nodes == 1: # ノードが1つの場合、隣接ノードは存在しない
            k_to_select = 0
        else:
            k_to_select = min(self.k_neighbors, num_nodes - 1)

        if k_to_select == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        # 全ての可能なエッジに対してランダムなスコアを生成 (num_nodes x num_nodes 行列)
        # この部分がメモリを大量に消費する可能性があります。
        rand_values = torch.rand(num_nodes, num_nodes, device=device)

        # 自己ループを防ぐため、対角成分を低い値(-1.0)に設定
        # torch.rand は [0, 1) の範囲の値を生成するため、-1.0 は選択されない
        rand_values.fill_diagonal_(-1.0)

        # 各ノードに対して、スコア上位 k_to_select 個のノードのインデックスを取得
        top_k_indices = torch.topk(rand_values, k_to_select, dim=1).indices

        # ソースノードのリストを作成: [0,0,...,0, 1,1,...,1, ..., N-1,...,N-1]
        # 各ノードiがk_to_select回繰り返される
        source_nodes = torch.arange(num_nodes, device=device).repeat_interleave(k_to_select)

        # ターゲットノードのリストを作成 (top_k_indicesをフラット化)
        target_nodes = top_k_indices.flatten()

        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        return edge_index

    def forward(self, x, edge_index=None):
        # エンコーダ
        # print("x: ", x.shape)
        x = self.encoder(x)
        # print("encoder out: ", x.shape)
        # エンコーダ
        x = x.unsqueeze(dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # ボトルネック（RelNet）
        batch_size, channels, height, width = x4.size()
        x4_reshaped = x4.view(batch_size, channels, -1).permute(0, 2, 1)
        x4_reshaped = x4_reshaped.reshape(-1, channels)

        if edge_index is None:
            # スパースグラフを作成
            num_nodes = x4_reshaped.size(0)
            edge_index = self.create_sparse_graph(num_nodes)  # GCNのとき
            # edge_index = num_nodes  # GCN不使用

        x4_processed = self.relnet(x4_reshaped, edge_index)
        x4_processed = x4_processed.view(batch_size, height, width, channels).permute(0, 3, 1, 2)

        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)
        logits = self.outc(d1)
        # print("x: ", x.shape)
        # マスクの適用
        out = x * logits
        out = out.squeeze()
        # デコーダ
        out = self.decoder(out)
        return out


def print_model_summary(model, batch_size, channels, length):
    # サンプル入力データを作成
    x = torch.randn(batch_size, channels, length).to(device)

    # モデルのサマリーを表示
    print("\nURelNet Model Summary:")
    summary(model, input_data=x)


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
    print("main")
    # サンプルデータの作成（入力サイズを縮小）
    batch = 1  # const.BATCHSIZE
    num_mic = 1  # 入力サイズを縮小
    length = 128000  # 入力サイズを縮小

    # ランダムな入力データを作成
    x = torch.randn(batch, num_mic, length).to(device)

    # モデルの初期化とデバイスへの移動
    model = URelNet(n_channels=num_mic, n_classes=num_mic, k_neighbors=8).to(device)

    # モデルのサマリーを表示
    # print_model_summary(model, batch, num_mic, length)

    # フォワードパス
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == '__main__':
    main()