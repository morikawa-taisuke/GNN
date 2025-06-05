import torch
from torch import nn
import os
from torchinfo import summary
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.contrib import tenumerate


from mymodule import const
import datasetClass


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


class RelNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, relation_dim, output_dim):
        super().__init__()
        self.relation_model = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, relation_dim),
            nn.ReLU()
        )
        self.output_model = nn.Sequential(
            nn.Linear(relation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, node_feats=8):
        N = node_feats.size(0)
        relation_vectors = []

        for i in range(N):
            for j in range(N):
                pair_feat = torch.cat([node_feats[i], node_feats[j]], dim=0)
                r_ij = self.relation_model(pair_feat)
                relation_vectors.append(r_ij)

        relation_vectors = torch.stack(relation_vectors, dim=0)  # shape: [N*N, D]
        R = relation_vectors.mean(dim=0)  # shape: [D]
        out = self.output_model(R)  # shape: [output_dim]
        return out


class URelNet2(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_dim=32, k_neighbors=8):
        super(URelNet2, self).__init__()
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
        self.relnet = RelNet(512, hidden_dim, hidden_dim, 512)

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
            neighbors = torch.randperm(num_nodes - 1, device=device)[:self.k_neighbors]
            neighbors[neighbors >= i] += 1  # 自分自身をスキップ
            edge_index[0, i * self.k_neighbors:(i + 1) * self.k_neighbors] = i
            edge_index[1, i * self.k_neighbors:(i + 1) * self.k_neighbors] = neighbors
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

        return x4_reshaped.size(0)


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
    dataset_path = f"{const.DATASET_DIR}/DEMAND_1ch/condition_4/noise_reverbe"
    out_path: str = "./RESULT/pth/result.pth"
    loss_func:str = "SISDR"
    batchsize:int = const.BATCHSIZE
    csv_path = f"{const.DATASET_DIR}/DEMAND_1ch/condition_4/condition4_data_length.csv"

    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUが使えれば使う

    with open(csv_path, "w") as csv_file:  # ファイルオープン
        csv_file.write(f"data_idx,data_length\n")

    """ Load dataset データセットの読み込み """
    # dataset = datasetClass.TasNet_dataset_csv(args.dataset, channel=channel, device=device) # データセットの読み込み
    dataset = datasetClass.TasNet_dataset2(dataset_path)  # データセットの読み込み
    dataset_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)

    """ ネットワークの生成 """
    model = URelNet2(n_channels=1, n_classes=1, k_neighbors=8).to(device)
    # model = U_Net().to(device)
    # print(f"\nmodel:{model}\n")                           # モデルのアーキテクチャの出力
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizerを選択(Adam)
    if loss_func != "SISDR":
        loss_function = nn.MSELoss().to(device)  # 損失関数に使用する式の指定(最小二乗誤差)

    # model.train()  # 学習モードに設定

    for _, (mix_data, target_data, idx) in tenumerate(dataset_loader):
        """ モデルの読み込み """
        mix_data, target_data = mix_data.to(device), target_data.to(device)  # データをGPUに移動

        """ 勾配のリセット """
        # optimizer.zero_grad()  # optimizerの初期化

        """ データの整形 """
        mix_data = mix_data.to(torch.float32)  # target_dataのタイプを変換 int16→float32
        target_data = target_data.to(torch.float32)  # target_dataのタイプを変換 int16→float32
        mix_data = mix_data.unsqueeze(dim=0)  # [バッチサイズ, マイク数，音声長]
        # target_data = target_data[np.newaxis, :, :] # 次元を増やす[1,音声長]→[1,1,音声長]
        # print("mix:", mix_data.shape)

        """ モデルに通す(予測値の計算) """
        estimate_data = model(mix_data)  # モデルに通す

        with open(csv_path, "a") as out_file:  # ファイルオープン
            out_file.write(f"{idx}, {estimate_data}\n")  # 書き込み
        # torch.cuda.empty_cache()    # メモリの解放 1epochごとに解放-



if __name__ == '__main__':
    main()