import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import knn_graph
from torchinfo import summary

from models.graph_utils import GraphBuilder, GraphConfig, NodeSelectionType, EdgeSelectionType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv1D(nn.Module):
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
        return self.double_conv(x)


class Down1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool1d(2), DoubleConv1D(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
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


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout_rate=0.5):
        super(GAT, self).__init__()
        self.heads = heads
        self.dropout_rate = dropout_rate

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_rate)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout_rate)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class UGNN(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        hidden_dim_gnn=32,
        num_node=8,
        gnn_type="GCN",
        gnn_heads=4,
        gnn_dropout=0.6,
        graph_config=None,
    ):
        super(UGNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_node_gnn = num_node
        self.gnn_type = gnn_type

        # エンコーダ
        self.encoder_dim = 512
        self.sampling_rate = 16000
        self.win = 4
        self.win_samples = int(self.sampling_rate * self.win / 1000)
        self.stride_samples = self.win_samples // 2

        self.encoder = nn.Conv1d(
            in_channels=n_channels,
            out_channels=self.encoder_dim,
            kernel_size=self.win_samples,
            bias=False,
            stride=self.stride_samples,
        )

        # U-Net (1D)
        self.inc = DoubleConv1D(self.encoder_dim, 64)
        self.down1 = Down1D(64, 128)
        self.down2 = Down1D(128, 256)
        self.down3 = Down1D(256, 512)

        # ボトルネックのGNN
        if gnn_type == "GCN":
            self.gnn = GCN(512, hidden_dim_gnn, 512)
        elif gnn_type == "GAT":
            self.gnn = GAT(512, hidden_dim_gnn, 512, heads=gnn_heads, dropout_rate=gnn_dropout)

        # デコーダパス
        self.up1 = Up1D(512, 256)
        self.up2 = Up1D(256, 128)
        self.up3 = Up1D(128, 64)
        self.outc = nn.Sequential(nn.Conv1d(64, n_classes, kernel_size=1), nn.Sigmoid())

        # デコーダ
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.encoder_dim,
            out_channels=n_channels,
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

    def create_knn_graph(self, x_nodes_batched, k, batch_size, num_nodes_per_sample):
        batch_indices = torch.arange(batch_size, device=x_nodes_batched.device).repeat_interleave(num_nodes_per_sample)
        edge_index = knn_graph(x=x_nodes_batched, k=k, batch=batch_indices, loop=False)
        return edge_index

    def forward(self, x):
        # エンコーダ
        x_encoded = self.encoder(x)

        # U-Net エンコーダパス
        x1 = self.inc(x_encoded)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # ボトルネックのGNN処理
        batch_size, channels, length = x4.size()
        x4_nodes = x4.permute(0, 2, 1).reshape(-1, channels)

        # k-NNグラフの作成
        edge_index = self.graph_builder.create_batch_graph(x4_nodes, batch_size, length)

        # GNN処理
        x4_processed = self.gnn(x4_nodes, edge_index)
        x4_processed = x4_processed.view(batch_size, length, channels).permute(0, 2, 1)

        # U-Net デコーダパス
        x = self.up1(x4_processed, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # マスク生成
        mask = self.outc(x)

        # マスク適用
        masked_features = x_encoded * mask

        # デコーダで波形に変換
        out = self.decoder(masked_features)

        return out


# def print_model_summary(model, batch_size=2, channels=1, length=16000):
#     x = torch.randn(batch_size, channels, length).to(device)
#     model = model.to(device)
#     print(f"\nModel Summary for {model.__class__.__name__}:")
#     print(f"Input shape: {x.shape}")
#     output = model(x)
#     print(f"Output shape: {output.shape}")


def print_model_summary(model, batch_size, channels, length):
    # サンプル入力データを作成
    x = torch.randn(batch_size, channels, length).to(device)

    # モデルのサマリーを表示
    print(f"\n{model.__class__.__name__} Model Summary:")
    summary(model, input_data=x)


def main():
    print("GNN.py main execution")
    # サンプルデータの作成（入力サイズを縮小）
    batch = 1
    num_mic = 1
    length = 16000 * 8  # 2秒の音声データ (例)

    # ランダムな入力データを作成
    # x = torch.randn(batch, num_mic, length).to(device)

    print("\n--- UGCN (Random Graph) ---")
    ugcn_model = UGNN(n_channels=num_mic, n_classes=1, num_node=8, gnn_type="GCN").to(device)
    print_model_summary(ugcn_model, batch, num_mic, length)
    x_ugcn = torch.randn(batch, num_mic, length).to(device)
    output_ugcn = ugcn_model(x_ugcn)
    print(f"UGCN Input shape: {x_ugcn.shape}, Output shape: {output_ugcn.shape}")

    print("\n--- UGAT (Random Graph, GAT in bottleneck) ---")
    ugat_model = UGNN(n_channels=num_mic, n_classes=1, num_node=8, gnn_type="GAT", gnn_heads=4, gnn_dropout=0.6).to(
        device
    )
    print_model_summary(ugat_model, batch, num_mic, length)
    x_ugat = torch.randn(batch, num_mic, length).to(device)
    output_ugat = ugat_model(x_ugat)  # forwardはUGCNNetのものを継承
    print(f"UGAT Input shape: {x_ugat.shape}, Output shape: {output_ugat.shape}")

    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage (after initializations):")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    # モデルのテスト
    main()
