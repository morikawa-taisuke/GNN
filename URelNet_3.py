import torch
from torch import nn
import torch.nn.functional as F
import os
from torchinfo import summary



# CUDAのメモリ管理設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAの可用性をチェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- RelNet モジュール ----------
class RelNet(nn.Module):
    def __init__(self, in_dim, g_hidden_dim=128, g_out_dim=64, f_hidden_dim=128, f_out_dim=None):
        super(RelNet, self).__init__()
        if f_out_dim is None:
            f_out_dim = in_dim
        self.g_theta = nn.Sequential(
            nn.Linear(in_dim * 2, g_hidden_dim),
            nn.ReLU(),
            nn.Linear(g_hidden_dim, g_out_dim),
            nn.ReLU()
        )
        self.f_phi = nn.Sequential(
            nn.Linear(g_out_dim, f_hidden_dim),
            nn.ReLU(),
            nn.Linear(f_hidden_dim, f_out_dim)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W

        # ノード化: (B, N, C)
        x_flat = x.view(B, C, N).permute(0, 2, 1)

        # 全ペア: (B, N, N, 2C)
        xi = x_flat.unsqueeze(2).expand(-1, -1, N, -1)
        xj = x_flat.unsqueeze(1).expand(-1, N, -1, -1)
        x_pairs = torch.cat([xi, xj], dim=-1)

        # g_theta: (B*N*N, 2C) → (B*N*N, Dg)
        g_input = x_pairs.view(B * N * N, -1)
        g_output = self.g_theta(g_input)

        # 集約: (B, Dg)
        g_output = g_output.view(B, N * N, -1).sum(dim=1)

        # f_phi: (B, Dg) → (B, Df)
        out = self.f_phi(g_output)

        # 空間次元に復元
        out = out.view(B, -1, 1, 1)
        out = out.expand(B, -1, H, W)
        return out

# ---------- U-Net モジュール ----------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # サイズ調整
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ---------- RelNet付きU-Net (URelNet) ----------

class URelNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(URelNet, self).__init__()
        self.encoder_dim = 512
        self.sampling_rate = 16000
        self.win = 4
        self.win = int(self.sampling_rate * self.win / 1000)
        self.stride = self.win // 2  # 畳み込み処理におけるフィルタが移動する幅
        # エンコーダ
        self.encoder = nn.Conv1d(in_channels=in_channels,  # 入力データの次元数 #=1もともとのやつ
                                 out_channels=self.encoder_dim,  # 出力データの次元数
                                 kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長なの?)
                                 bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
                                 stride=self.stride)  # 畳み込み処理の移動幅

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.rel_net = RelNet(in_dim=512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, in_channels)

        # デコーダ
        self.decoder = nn.ConvTranspose1d(in_channels=self.encoder_dim,  # 入力次元数
                                          out_channels=out_channels,  # 出力次元数 1もともとのやつ
                                          kernel_size=self.win,  # カーネルサイズ
                                          bias=False,
                                          stride=self.stride)  # 畳み込み処理の移動幅

    def forward(self, x):
        x = self.encoder(x)
        print("encoder out: ", x.shape)
        # エンコーダ
        x = x.unsqueeze(dim=1)
        print("inc_input: ",x.shape)
        x1 = self.inc(x)     # (B, 64, H, W)
        print("inc_output: ",x.shape)
        x2 = self.down1(x1)  # (B, 128, H/2, W/2)
        print("down1_output: ",x.shape)
        x3 = self.down2(x2)  # (B, 256, H/4, W/4)
        print("down2_output: ",x.shape)
        x4 = self.down3(x3)  # (B, 512, H/8, W/8)
        print("down3_output: ",x.shape)

        x4 = self.rel_net(x4)  # RelNetでボトルネック強化
        print("rel_output: ",x.shape)

        x = self.up1(x4, x3)
        print("up1_output: ",x.shape)
        x = self.up2(x, x2)
        print("up2_output: ",x.shape)
        x = self.up3(x, x1)
        print("up3_output: ",x.shape)
        logits = self.outc(x)
        print("outc_output: ",logits.shape)
        logits = logits.squeeze()
        # デコーダ
        print("decoder_input: ",logits.shape)
        logits = self.decoder(logits)
        print("decoder_output: ",logits.shape)

        return logits



def print_model_summary(model, batch_size, channels, length):
    # サンプル入力データを作成
    x = torch.randn(batch_size, channels, length).to(device)

    # モデルのサマリーを表示
    print("\nURelNet Model Summary:")
    summary(model, input_data=x)


def main():
    print("main")
    # サンプルデータの作成（入力サイズを縮小）
    batch = 1  # const.BATCHSIZE
    num_mic = 1  # 入力サイズを縮小
    length = 8000  # 入力サイズを縮小

    # ランダムな入力データを作成
    x = torch.randn(batch, num_mic, length).to(device)

    # モデルの初期化とデバイスへの移動
    model = URelNet(in_channels=1, out_channels=1).to(device)

    # モデルのサマリーを表示
    print_model_summary(model, batch, num_mic, length)

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