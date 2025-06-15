import torch
import torch.nn as nn
import os
from torchinfo import summary
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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


class UGCN(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_dim=32, k_neighbors=8):
        super(UGCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.k_neighbors = k_neighbors

        self.encoder_dim=512
        self.sampling_rate=16000
        self.win = 4
        self.win=int(self.sampling_rate * self.win / 1000)
        self.stride = self.win // 2   # 畳み込み処理におけるフィルタが移動する幅

        # エンコーダ
        self.encoder = nn.Conv1d(in_channels=n_channels,    # 入力データの次元数 #=1もともとのやつ
                                 out_channels=self.encoder_dim, # 出力データの次元数
                                 kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長なの?)
                                 bias=False,    # バイアスの有無(出力に学習可能なバイアスの追加)
                                 stride=self.stride)    # 畳み込み処理の移動幅
        # デコーダ
        self.decoder = nn.ConvTranspose1d(in_channels = self.encoder_dim,   # 入力次元数
                                          out_channels=n_channels,  # 出力次元数 1もともとのやつ
                                          kernel_size= self.win,    # カーネルサイズ
                                          bias=False,
                                          stride=self.stride)   # 畳み込み処理の移動幅


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
        # 各ノードに対してk個の近傍ノードを選択
        edge_index = torch.zeros((2, num_nodes * self.k_neighbors), dtype=torch.long, device=device)
        for i in range(num_nodes):
            # ランダムにk個の近傍を選択（自分自身は除外）
            neighbors = torch.randperm(num_nodes-1, device=device)[:self.k_neighbors]
            neighbors[neighbors >= i] += 1  # 自分自身をスキップ
            edge_index[0, i*self.k_neighbors:(i+1)*self.k_neighbors] = i
            edge_index[1, i*self.k_neighbors:(i+1)*self.k_neighbors] = neighbors
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
            edge_index = self.create_sparse_graph(num_nodes)
        
        x4_processed = self.relnet(x4_reshaped, edge_index)
        x4_processed = x4_processed.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # デコーダー
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

def main():
    print("main")
    # サンプルデータの作成（入力サイズを縮小）
    batch = 1 # const.BATCHSIZE
    num_mic = 1  # 入力サイズを縮小
    length = 128000  # 入力サイズを縮小
    
    # ランダムな入力画像を作成
    x = torch.randn(batch, num_mic, length).to(device)

    # モデルの初期化とデバイスへの移動
    model = UGCN(n_channels=num_mic, n_classes=num_mic, k_neighbors=8).to(device)
    
    # モデルのサマリーを表示
    print_model_summary(model, batch, num_mic, length)
    
    # フォワードパス
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

if __name__ == '__main__':
    main()