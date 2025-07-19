import torch
from torch import nn
import os
from torchinfo import summary
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv  # GATConv をインポート
from torch_geometric.nn import knn_graph


# CUDAのメモリ管理設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CUDAの可用性をチェック
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
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

        # マルチヘッドアテンションを考慮して、中間層の出力次元は hidden_dim * heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
        # 次のGATConvの入力次元は hidden_dim * heads
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_rate)
        # 最終層の出力はoutput_dimで、通常heads=1 (concat=Falseに相当)
        # または、headsを維持して後で平均化するなら concat=True のまま heads を指定
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout_rate)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index)  # 最終層の出力はそのまま（softmaxなどはモデル全体の後段で）
        return x


class UGCNNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, hidden_dim=32, num_node=8):
        super(UGCNNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_node = num_node

        self.encoder_dim = 512
        self.sampling_rate = 16000
        self.win = 4
        self.win = int(self.sampling_rate * self.win / 1000)
        self.stride = self.win // 2  # 畳み込み処理におけるフィルタが移動する幅

        # エンコーダ
        self.encoder = nn.Conv1d(
            in_channels=n_channels,  # 入力データの次元数 #=1もともとのやつ
            out_channels=self.encoder_dim,  # 出力データの次元数
            kernel_size=self.win,  # 畳み込みのサイズ(波形領域なので窓長なの?)
            bias=False,  # バイアスの有無(出力に学習可能なバイアスの追加)
            stride=self.stride,
        )  # 畳み込み処理の移動幅
        # デコーダ
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.encoder_dim,  # 入力次元数
            out_channels=n_channels,  # 出力次元数 1もともとのやつ
            kernel_size=self.win,  # カーネルサイズ
            bias=False,
            stride=self.stride,
        )  # 畳み込み処理の移動幅

        # エンコーダー部分
        self.inc = DoubleConv(n_channels, 64)  # U-Netの入力は1チャンネルの特徴マップを想定
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # ボトルネック部分のGNN
        self.gnn = GCN(512, hidden_dim, 512)  # GNNの入力はU-Netボトルネックのチャネル数

        # デコーダー部分
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)  # マスク生成

    def create_graph(self, num_nodes):
        """
        ノードごとにランダムにk個の異なる隣接ノードを選択してスパースグラフを作成します。
        自己ループは作成されません。
        この最適化版では、torch.topkを使用して処理をベクトル化し、Pythonループを削減します。
        注意: num_nodesが大きい場合、num_nodes x num_nodes の行列を一時的に使用するため、
              メモリ使用量が増加する可能性があります。
        """
        if num_nodes == 0 or self.num_node == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        # 実際に選択する隣接ノードの数 (num_nodes - 1 を超えることはない)
        if num_nodes == 1:  # ノードが1つの場合、隣接ノードは存在しない
            k_to_select = 0
        else:
            k_to_select = min(self.num_node, num_nodes - 1)

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
        # x: [B, C_in, L_in]
        x_encoded = self.encoder(x)  # [B, encoder_dim, L_encoded]

        # U-Net入力形式へ変換
        # [B, encoder_dim, L_encoded] -> [B, 1, encoder_dim, L_encoded] (H=encoder_dim, W=L_encodedと解釈)
        x_unet_input = x_encoded.unsqueeze(dim=1)

        # U-Net エンコーダ
        x1 = self.inc(x_unet_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # [B, 512, H_bottleneck, W_bottleneck]

        # ボトルネック（GCN）
        batch_size, channels_bottleneck, height_bottleneck, width_bottleneck = x4.size()
        # GCN入力形式へ変換: [B, C, H, W] -> [B, H*W, C] -> [B*H*W, C] (ノードリスト)
        x4_reshaped = x4.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)

        if edge_index is None:
            # スパースグラフを作成
            # 注意: バッチ処理の場合、グラフの作り方を工夫する必要があるかもしれません。
            #       例えば、バッチ内の各サンプルごとにグラフを作成し、それらを結合するなど。
            #       現在の実装は、バッチ内の全ノードを1つのグラフとして扱います。
            num_nodes = x4_reshaped.size(0)
            edge_index = self.create_graph(num_nodes)

        x4_processed_flat = self.gnn(x4_reshaped, edge_index)  # [B*H*W, C_out_gnn]

        # U-Net形式へ戻す: [B*H*W, C_out_gcn] -> [B, H, W, C_out_gcn] -> [B, C_out_gcn, H, W]
        x4_processed = x4_processed_flat.view(batch_size, height_bottleneck, width_bottleneck, channels_bottleneck).permute(
            0, 3, 1, 2
        )

        # U-Net デコーダ
        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)
        mask_pred = torch.sigmoid(self.outc(d1))  # [B, n_classes, H_mask, W_mask] (通常 H_mask=encoder_dim, W_mask=L_encoded)

        # マスクをx_encodedの次元に合わせる (n_classes=1を想定)
        if mask_pred.size(2) != x_encoded.size(1) or mask_pred.size(3) != x_encoded.size(2):
            mask_pred_resized = F.interpolate(
                mask_pred, size=(x_encoded.size(1), x_encoded.size(2)), mode="bilinear", align_corners=False
            )
        else:
            mask_pred_resized = mask_pred

        # マスクの適用
        # x_encoded: [B, encoder_dim, L_encoded]
        # mask_pred_resized: [B, 1, encoder_dim, L_encoded]
        # マスクをブロードキャストするために次元を調整
        if self.n_classes == 1:
            # [B, 1, encoder_dim, L_encoded] -> [B, encoder_dim, L_encoded] (encoder_dim次元にマスクを適用するため)
            # この部分のマスクの適用方法はモデルの意図によります。
            # URelNetでは mask_upsampled を [B, n_classes, L_encoded] にして x_encoded にかけていた。
            # ここでは mask_pred_resized が [B, 1, H_enc, W_enc] で、x_encoded が [B, H_enc, W_enc]
            # なので、mask_pred_resized.squeeze(1) で [B, H_enc, W_enc] にして要素積。
            masked_x_encoded = x_encoded * mask_pred_resized.squeeze(1)
        else:
            # n_classes > 1 の場合の処理 (例: 各クラスが特定の周波数帯のマスクなど)
            # ここでは単純な要素積を仮定
            masked_x_encoded = x_encoded * mask_pred_resized  # 要確認: n_classesとencoder_dimの関係

        # デコーダ
        out_waveform = self.decoder(masked_x_encoded)  # [B, C_out, L_out]
        return out_waveform


class UGATNet(UGCNNet):  # UGCNNetを継承
    def __init__(self, n_channels, n_classes=1, hidden_dim=32, num_node=8, gat_heads=8, gat_dropout=0.5):
        super(UGATNet, self).__init__(n_channels, n_classes, hidden_dim, num_node)

        # ボトルネック部分をGATに置き換える
        # GATの入力次元はU-Netエンコーダのボトルネック部分のチャネル数(512)
        # GATの出力次元も同様に512とする
        self.gnn = GAT(512, hidden_dim, 512, heads=gat_heads, dropout_rate=gat_dropout)

    # forwardメソッドはUGCNNetと同じものを利用できる（gnnがGATに変わるだけ）
    # create_sparse_graph もUGCNNetのものをそのまま利用（ランダムグラフ生成）
    # もしk-NNグラフなど他のグラフ生成方法を使いたい場合は、ここでオーバーライドするか、
    # UGCNNet2のように別のグラフ生成メソッドを定義してforward内で呼び出す。


class UGCNNet2(UGCNNet):
    def create_graph(self, x_nodes_batched, k, batch_size, num_nodes_per_sample):
        # x: [batch_size * num_nodes_per_sample, num_features]
        batch_indices = torch.arange(batch_size, device=x_nodes_batched.device).repeat_interleave(num_nodes_per_sample)
        edge_index = knn_graph(x=x_nodes_batched, k=k, batch=batch_indices, loop=False)  # 自己ループなし
        return edge_index

    def forward(self, x):  # edge_indexは内部で生成するため引数から削除
        # エンコーダ (波形 -> 潜在特徴量)
        # x: [batch_size, n_channels, length]
        x_encoded = self.encoder(x)  # [batch_size, encoder_dim, L_encoded]

        # 2. U-Netの入力形式に変換
        # U-NetはConv2dを使うため、[batch_size, channels, H, W]の形式が必要
        # x_encoded: [batch_size, encoder_dim, L_encoded]
        # H=encoder_dim, W=L_encodedと解釈されるように次元を追加
        x_unet_input = x_encoded.unsqueeze(dim=1)

        # 3. U-Net エンコーダー部分
        x1 = self.inc(x_unet_input)  # [B, 64, H1, W1]
        x2 = self.down1(x1)  # [B, 128, H2, W2]
        x3 = self.down2(x2)  # [B, 256, H3, W3]
        x4 = self.down3(x3)  # [B, 512, H_bottleneck, W_bottleneck]

        # ボトルネック（GCN or GAT）
        batch_size, channels_bottleneck, height_bottleneck, width_bottleneck = x4.size()
        x4_reshaped = x4.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)

        # k-NNグラフを動的に構築
        num_nodes_per_sample = height_bottleneck * width_bottleneck
        if num_nodes_per_sample > 0:
            edge_index = self.create_graph(x4_reshaped, self.num_node, batch_size, num_nodes_per_sample)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x4_reshaped.device)

        x4_processed_flat = self.gnn(x4_reshaped, edge_index)  # GNN (GCN or GAT)

        x4_processed = x4_processed_flat.view(batch_size, height_bottleneck, width_bottleneck, channels_bottleneck).permute(
            0, 3, 1, 2
        )

        # U-Net デコーダー
        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)
        mask_pred = self.outc(d1)
        mask_pred = torch.sigmoid(mask_pred)

        # --- ここからデバッグコード ---
        # print(f"--- Debug mask_pred in UGCNNet2 ---")
        # print(f"mask_pred.shape: {mask_pred.shape}")
        # if mask_pred.numel() > 0: # テンソルが空でないことを確認
        #     print(f"mask_pred.min(): {mask_pred.min().item()}")
        #     print(f"mask_pred.max(): {mask_pred.max().item()}")
        #     print(f"mask_pred.mean(): {mask_pred.mean().item()}")
        #     print(f"mask_pred.dtype: {mask_pred.dtype}")
        # --- ここまでデバッグコード ---

        if mask_pred.size(2) != x_encoded.size(1) or mask_pred.size(3) != x_encoded.size(2):
            mask_pred_resized = F.interpolate(
                mask_pred, size=(x_encoded.size(1), x_encoded.size(2)), mode="bilinear", align_corners=False
            )
        else:
            mask_pred_resized = mask_pred

        if self.n_classes == 1:
            masked_x_encoded = x_encoded * mask_pred_resized.squeeze(1)
        else:
            masked_x_encoded = x_encoded * mask_pred_resized

        output_waveform = self.decoder(masked_x_encoded)
        return output_waveform


class UGATNet2(UGCNNet2):  # UGCNNet2 を継承
    def __init__(self, n_channels, n_classes=1, hidden_dim=32, num_node=8, gat_heads=8, gat_dropout=0.5):
        # UGCNNet2 (親クラス) の __init__ を呼び出す
        # UGCNNet2 は UGCNNet を継承しており、UGCNNet で self.gnn が GCN で初期化される
        super(UGATNet2, self).__init__(n_channels, n_classes, hidden_dim, num_node)

        # ボトルネック部分のGNNをGATに置き換える
        # GATの入力次元はU-Netエンコーダのボトルネック部分のチャネル数(512)
        # GATの出力次元も同様に512とする
        self.gnn = GAT(512, hidden_dim, 512, heads=gat_heads, dropout_rate=gat_dropout)

    # forwardメソッドとcreate_knn_graph_for_batchメソッドはUGCNNet2のものをそのまま利用


def print_model_summary(model, batch_size, channels, length):
    # サンプル入力データを作成
    x = torch.randn(batch_size, channels, length).to(device)

    # モデルのサマリーを表示
    print(f"\n{model.__class__.__name__} Model Summary:")
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
    print("GCN.py main execution")
    # サンプルデータの作成（入力サイズを縮小）
    batch = 1
    num_mic = 1
    length = 16000 * 8  # 2秒の音声データ (例)

    # ランダムな入力データを作成
    # x = torch.randn(batch, num_mic, length).to(device)

    print("\n--- UGCNNet (Random Graph) ---")
    ugcn_model = UGCNNet(n_channels=num_mic, n_classes=1, num_node=8).to(device)
    print_model_summary(ugcn_model, batch, num_mic, length)
    x_ugcn = torch.randn(batch, num_mic, length).to(device)
    output_ugcn = ugcn_model(x_ugcn)
    print(f"UGCNNet Input shape: {x_ugcn.shape}, Output shape: {output_ugcn.shape}")

    print("\n--- UGATNet (Random Graph, GAT in bottleneck) ---")
    ugat_model = UGATNet(n_channels=num_mic, n_classes=1, num_node=8, gat_heads=4, gat_dropout=0.6).to(device)
    print_model_summary(ugat_model, batch, num_mic, length)
    x_ugat = torch.randn(batch, num_mic, length).to(device)
    output_ugat = ugat_model(x_ugat)  # forwardはUGCNNetのものを継承
    print(f"UGATNet Input shape: {x_ugat.shape}, Output shape: {output_ugat.shape}")

    print("\n--- UGCNNet2 (k-NN Graph, GCN in bottleneck) ---")
    ugcn2_model = UGCNNet2(n_channels=num_mic, n_classes=1, num_node=8).to(device)
    print_model_summary(ugcn2_model, batch, num_mic, length)
    x_ugcn2 = torch.randn(batch, num_mic, length).to(device)
    output_ugcn2 = ugcn2_model(x_ugcn2)
    print(f"UGCNNet2 Input shape: {x_ugcn2.shape}, Output shape: {output_ugcn2.shape}")

    print("\n--- UGATNet2 (k-NN Graph, GAT in bottleneck) ---")
    ugat2_model = UGATNet2(n_channels=num_mic, n_classes=1, num_node=8, gat_heads=4, gat_dropout=0.6).to(device)
    print_model_summary(ugat2_model, batch, num_mic, length)
    # x_ugat2 = torch.randn(batch, num_mic, length).to(device)
    # output_ugat2 = ugat2_model(x_ugat2) # forwardはUGCNNet2のものを継承
    # print(f"UGATNet2 Input shape: {x_ugat2.shape}, Output shape: {output_ugat2.shape}")

    # モデルのサマリーを表示
    # print_model_summary(model, batch, num_mic, length)

    # フォワードパス
    # output = model(x)
    # print(f"\nInput shape: {x.shape}")
    # print(f"Output shape: {output.shape}")

    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage (after initializations):")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    main()
