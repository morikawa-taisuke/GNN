import torch
from torch import nn
import os
from torchinfo import summary
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import knn_graph

# CUDAのメモリ管理設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAの可用性をチェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"SpeqGNN.py using device: {device}")


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


class SpeqGCNNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_dim=32, num_node=8, n_fft=512, hop_length=256, win_length=None):
        """
        Args:
            n_channels (int): Number of input channels in the time-frequency domain feature map.
                              For example, 1 for magnitude spectrogram, 2 for real and imaginary parts.
            n_classes (int): Number of output channels for the mask (typically 1).
            hidden_dim (int): Hidden dimension for GCN.
            num_node (int): Number of neighbors for k-NN graph construction.
            n_fft (int): FFT size for STFT/ISTFT.
            hop_length (int): Hop length for STFT/ISTFT.
            win_length (int): Window length for STFT/ISTFT. If None, defaults to n_fft.
        """
        super(SpeqGCNNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_node = num_node

        # ISTFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        # Create window tensor, will be moved to device in forward pass
        self.window = torch.hann_window(self.win_length)
        
        # U-Net Encoder part
        # The input `x` is already a time-frequency representation [B, C_feat, F, T]
        # So, self.inc takes n_channels (C_feat) as input.
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512) # Bottleneck channel size is 512

        # Bottleneck GNN
        self.gnn = GCN(512, hidden_dim, 512) # GNN input/output matches bottleneck channels

        # U-Net Decoder part
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0) # Mask generation

    def create_graph(self, num_nodes):
        """
        ノードごとにランダムにk個の異なる隣接ノードを選択してスパースグラフを作成します。
        自己ループは作成されません。
        この最適化版では、torch.topkを使用して処理をベクトル化し、Pythonループを削減します。
        注意: num_nodesが大きい場合、num_nodes x num_nodes の行列を一時的に使用するため、
              メモリ使用量が増加する可能性があります。
        """
        if num_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        if self.num_node == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        # 実際に選択する隣接ノードの数 (num_nodes - 1 を超えることはない)
        if num_nodes == 1: # ノードが1つの場合、隣接ノードは存在しない
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

    def forward(self, x_magnitude, complex_spec_input, original_length=None, edge_index=None):
        """
        Args:
            x_magnitude (torch.Tensor): Input magnitude spectrogram [Batch, n_channels, FreqBins, TimeFrames].
            complex_spec_input (torch.Tensor): Complex spectrogram of the original input mixture
                                               [Batch, FreqBins, TimeFrames] (assuming mono audio input to STFT).
            original_length (int, optional): Length of the original time-domain signal.
                                             Used for ISTFT length parameter. Defaults to None.
        """
        input_freq_bins = x_magnitude.size(2)
        input_time_frames = x_magnitude.size(3)

        # 1. U-Net Encoder
        x1 = self.inc(x_magnitude)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # Bottleneck: [B, 512, F_bottle, T_bottle]

        # 2. GNN in Bottleneck
        batch_size, channels_bottleneck, height_bottleneck, width_bottleneck = x4.size()
        # Reshape for GNN: [B, C, H, W] -> [B, H*W, C] -> [B*H*W, C]
        x4_reshaped = x4.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)

        if edge_index is None:
            # スパースグラフを作成
            # 注意: バッチ処理の場合、グラフの作り方を工夫する必要があるかもしれません。
            #       例えば、バッチ内の各サンプルごとにグラフを作成し、それらを結合するなど。
            #       現在の実装は、バッチ内の全ノードを1つのグラフとして扱います。
            num_nodes = x4_reshaped.size(0)
            edge_index = self.create_graph(num_nodes)

        x4_processed_flat = self.gnn(x4_reshaped, edge_index)

        # Reshape back to U-Net format: [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        x4_processed = x4_processed_flat.view(batch_size, height_bottleneck, width_bottleneck, channels_bottleneck).permute(0, 3, 1, 2)

        # 3. U-Net Decoder
        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        # 4. Mask Prediction
        mask_pred_raw = self.outc(d1) # [B, n_classes, F_mask, T_mask]
        mask_pred = torch.sigmoid(mask_pred_raw)

        # 5. Resize mask to match input feature map dimensions (F, T)
        if mask_pred.size(2) != input_freq_bins or mask_pred.size(3) != input_time_frames:
            mask_pred_resized = F.interpolate(mask_pred, size=(input_freq_bins, input_time_frames), mode='bilinear', align_corners=False)
        else:
            mask_pred_resized = mask_pred

        # 6. Apply mask
        # x_magnitude: [B, n_channels, F, T]
        # mask_pred_resized: [B, n_classes, F, T]
        # If n_channels == n_classes, direct multiplication.
        # If n_classes == 1, mask is broadcasted across input channels.
        # Assuming n_channels (of x_magnitude) is 1 if n_classes is 1 for the mask.
        predicted_magnitude_tf = x_magnitude * mask_pred_resized # [B, n_channels(or n_classes), F, T]

        # 7. ISTFT
        # Assuming n_classes = 1 for the mask, so predicted_magnitude_tf is [B, 1, F, T]
        # And complex_spec_input is [B, F, T]
        if predicted_magnitude_tf.size(1) == 1:
            predicted_magnitude_for_istft = predicted_magnitude_tf.squeeze(1) # [B, F, T]
        else:
            # If n_classes > 1, this part needs specific handling.
            # For now, assume the first channel is the target magnitude.
            print(f"Warning: SpeqGCNNet.forward - n_classes > 1 ({predicted_magnitude_tf.size(1)}), using the first channel for ISTFT.")
            predicted_magnitude_for_istft = predicted_magnitude_tf[:, 0, :, :]

        # Ensure complex_spec_input has the same F, T dimensions as predicted_magnitude_for_istft
        # This should hold if mask_pred_resized was interpolated to input_freq_bins, input_time_frames
        phase = torch.angle(complex_spec_input) # [B, F, T]

        # Reconstruct complex spectrogram using predicted magnitude and input phase
        reconstructed_complex_spec = torch.polar(predicted_magnitude_for_istft, phase) # [B, F, T]

        output_waveform = torch.istft(reconstructed_complex_spec,
                                      n_fft=self.n_fft,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length,
                                      window=self.window.to(reconstructed_complex_spec.device),
                                      return_complex=False,
                                      length=original_length) # [B, L_output]
        return output_waveform


class SpeqGATNet(SpeqGCNNet):
    def __init__(self, n_channels, n_classes, hidden_dim=32, num_node=8, gat_heads=8, gat_dropout=0.5,
                 n_fft=512, hop_length=256, win_length=None):
        super(SpeqGATNet, self).__init__(n_channels, n_classes, hidden_dim, num_node, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # Override the GNN part with GAT
        self.gnn = GAT(512, hidden_dim, 512, heads=gat_heads, dropout_rate=gat_dropout)
    # forward and create_graph are inherited from SpeqGCNNet


class SpeqGCNNet2(SpeqGCNNet):
    def create_graph(self, x_nodes_batched, k, batch_size, num_nodes_per_sample):
        """ Create a k-NN graph for the given node features.
        Args:
            x_nodes_batched (torch.Tensor): Node features of shape [batch_size * num_nodes_per_sample, num_features].
            k (int): Number of nearest neighbors to connect.
            batch_size (int): Number of samples in the batch.
            num_nodes_per_sample (int): Number of nodes per sample in the batch.
        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges].
        """
        # x: [batch_size * num_nodes_per_sample, num_features]
        batch_indices = torch.arange(batch_size, device=x_nodes_batched.device).repeat_interleave(num_nodes_per_sample)
        edge_index = knn_graph(x=x_nodes_batched, k=k, batch=batch_indices, loop=False) # 自己ループなし
        return edge_index
    
    def forward(self, x_magnitude, complex_spec_input, original_length=None, edge_index=None):
        """
        Args:
            x_magnitude (torch.Tensor): Input magnitude spectrogram [Batch, n_channels, FreqBins, TimeFrames].
            complex_spec_input (torch.Tensor): Complex spectrogram of the original input mixture
                                            [Batch, FreqBins, TimeFrames] (assuming mono audio input to STFT).
            original_length (int, optional): Length of the original time-domain signal.
                                            Used for ISTFT length parameter. Defaults to None.
        """
        input_freq_bins = x_magnitude.size(2)
        input_time_frames = x_magnitude.size(3)

        # 1. U-Net Encoder
        x1 = self.inc(x_magnitude)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # Bottleneck: [B, 512, F_bottle, T_bottle]

        # 2. GNN in Bottleneck
        batch_size, channels_bottleneck, height_bottleneck, width_bottleneck = x4.size()
        # Reshape for GNN: [B, C, H, W] -> [B, H*W, C] -> [B*H*W, C]
        x4_reshaped = x4.view(batch_size, channels_bottleneck, -1).permute(0, 2, 1).reshape(-1, channels_bottleneck)

        num_nodes_per_sample = height_bottleneck * width_bottleneck
        if num_nodes_per_sample > 0 :
            edge_index = self.create_graph(x4_reshaped, self.num_node, batch_size, num_nodes_per_sample)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long, device=x4_reshaped.device)

        x4_processed_flat = self.gnn(x4_reshaped, edge_index)

        # Reshape back to U-Net format: [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        x4_processed = x4_processed_flat.view(batch_size, height_bottleneck, width_bottleneck, channels_bottleneck).permute(0, 3, 1, 2)

        # 3. U-Net Decoder
        d3 = self.up1(x4_processed, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up3(d2, x1)

        # 4. Mask Prediction
        mask_pred_raw = self.outc(d1) # [B, n_classes, F_mask, T_mask]
        mask_pred = torch.sigmoid(mask_pred_raw)

        # 5. Resize mask to match input feature map dimensions (F, T)
        if mask_pred.size(2) != input_freq_bins or mask_pred.size(3) != input_time_frames:
            mask_pred_resized = F.interpolate(mask_pred, size=(input_freq_bins, input_time_frames), mode='bilinear', align_corners=False)
        else:
            mask_pred_resized = mask_pred

        # 6. Apply mask
        # x_magnitude: [B, n_channels, F, T]
        # mask_pred_resized: [B, n_classes, F, T]
        # If n_channels == n_classes, direct multiplication.
        # If n_classes == 1, mask is broadcasted across input channels.
        # Assuming n_channels (of x_magnitude) is 1 if n_classes is 1 for the mask.
        predicted_magnitude_tf = x_magnitude * mask_pred_resized # [B, n_channels(or n_classes), F, T]

        # 7. ISTFT
        # Assuming n_classes = 1 for the mask, so predicted_magnitude_tf is [B, 1, F, T]
        # And complex_spec_input is [B, F, T]
        if predicted_magnitude_tf.size(1) == 1:
            predicted_magnitude_for_istft = predicted_magnitude_tf.squeeze(1) # [B, F, T]
        else:
            # If n_classes > 1, this part needs specific handling.
            # For now, assume the first channel is the target magnitude.
            print(f"Warning: SpeqGCNNet.forward - n_classes > 1 ({predicted_magnitude_tf.size(1)}), using the first channel for ISTFT.")
            predicted_magnitude_for_istft = predicted_magnitude_tf[:, 0, :, :]

        # Ensure complex_spec_input has the same F, T dimensions as predicted_magnitude_for_istft
        # This should hold if mask_pred_resized was interpolated to input_freq_bins, input_time_frames
        phase = torch.angle(complex_spec_input) # [B, F, T]

        # Reconstruct complex spectrogram using predicted magnitude and input phase
        reconstructed_complex_spec = torch.polar(predicted_magnitude_for_istft, phase) # [B, F, T]

        output_waveform = torch.istft(reconstructed_complex_spec,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length,
                                    window=self.window.to(reconstructed_complex_spec.device),
                                    return_complex=False,
                                    length=original_length) # [B, L_output]
        return output_waveform

class SpeqGATNet2(SpeqGCNNet2):
    def __init__(self, n_channels, n_classes, hidden_dim=32, num_node=8, gat_heads=8, gat_dropout=0.5, n_fft=512, hop_length=256, win_length=None):
        super(SpeqGATNet2, self).__init__(n_channels, n_classes, hidden_dim, num_node, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # Override the GNN part with GAT
        self.gnn = GAT(512, hidden_dim, 512, heads=gat_heads, dropout_rate=gat_dropout)
    # forward and create_graph are inherited from SpeqGCNNet2

def print_model_summary(model, batch_size, num_channels, freq_bins, time_frames):
    # Sample input data for time-frequency domain
    # For SpeqGCNNet, input is x_magnitude and complex_spec_input, and original_length
    x_mag = torch.randn(batch_size, num_channels, freq_bins, time_frames).to(device)
    # complex_spec_input should be [B, F, T]
    complex_spec = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.cfloat).to(device)
    original_len = time_frames * model.hop_length # Approximate
    print(f"\n{model.__class__.__name__} Model Summary:")
    # torchinfo summary might need adjustment for multiple inputs of different types
    # For simplicity, we'll just print the structure based on one input if direct summary fails.
    try:
        summary(model, input_data=(x_mag, complex_spec, original_len))
    except Exception as e:
        print(f"Could not generate summary with multiple inputs: {e}")
        summary(model, input_data=x_mag) # Fallback to magnitude input only for structure view


if __name__ == '__main__':
    print("SpeqGNN.py main execution")
    batch = 2
    num_input_channels = 1 # Example: Magnitude spectrogram
    n_fft_val = 512
    hop_length_val = 128
    num_freq_bins = n_fft_val // 2 + 1
    num_time_frames = 100  # Example: 100 time frames
    num_mask_classes = 1   # Typically 1 for a single enhancement mask
    # Approximate original audio length for testing ISTFT
    dummy_original_length = num_time_frames * hop_length_val

    print("\n--- SpeqGCNNet (k-NN Graph, GCN in bottleneck, Spectrogram I/O) ---")
    speq_gcn_model = SpeqGCNNet(n_channels=num_input_channels,
                                n_classes=num_mask_classes,
                                num_node=8,
                                n_fft=n_fft_val,
                                hop_length=hop_length_val).to(device)
    print_model_summary(speq_gcn_model, batch, num_input_channels, num_freq_bins, num_time_frames)
    
    # Test forward pass
    dummy_magnitude_spec = torch.randn(batch, num_input_channels, num_freq_bins, num_time_frames).to(device)
    dummy_complex_spec_input = torch.randn(batch, num_freq_bins, num_time_frames, dtype=torch.cfloat).to(device)
    
    output_waveform_gcn = speq_gcn_model(dummy_magnitude_spec, dummy_complex_spec_input, dummy_original_length)
    print(f"SpeqGCNNet Input magnitude shape: {dummy_magnitude_spec.shape}")
    print(f"SpeqGCNNet Input complex_spec shape: {dummy_complex_spec_input.shape}")
    print(f"SpeqGCNNet Output waveform shape: {output_waveform_gcn.shape}")
    assert output_waveform_gcn.ndim == 2 and output_waveform_gcn.size(0) == batch, "Output shape mismatch for SpeqGCNNet"
    # Length check can be approximate due to STFT/ISTFT padding/truncation
    # assert output_waveform_gcn.size(1) == dummy_original_length 

    print("\n--- SpeqGATNet (k-NN Graph, GAT in bottleneck, Spectrogram I/O, Waveform Output) ---")
    speq_gat_model = SpeqGATNet(n_channels=num_input_channels,
                                n_classes=num_mask_classes,
                                num_node=8,
                                gat_heads=4,
                                gat_dropout=0.6,
                                n_fft=n_fft_val,
                                hop_length=hop_length_val).to(device)
    print_model_summary(speq_gat_model, batch, num_input_channels, num_freq_bins, num_time_frames)
    # Test forward pass
    output_waveform_gat = speq_gat_model(dummy_magnitude_spec, dummy_complex_spec_input, dummy_original_length)
    print(f"SpeqGATNet Input magnitude shape: {dummy_magnitude_spec.shape}")
    print(f"SpeqGATNet Input complex_spec shape: {dummy_complex_spec_input.shape}")
    print(f"SpeqGATNet Output waveform shape: {output_waveform_gat.shape}")
    assert output_waveform_gat.ndim == 2 and output_waveform_gat.size(0) == batch, "Output shape mismatch for SpeqGATNet"

    print("\n--- SpeqGCNNet2 Model Summary ---")
    speq_gcn_model2 = SpeqGCNNet2(n_channels=num_input_channels,
                                  n_classes=num_mask_classes,
                                  num_node=8,
                                  n_fft=n_fft_val,
                                  hop_length=hop_length_val).to(device)
    print_model_summary(speq_gcn_model2, batch, num_input_channels, num_freq_bins, num_time_frames)
    # Test forward pass
    output_waveform_gcn2 = speq_gcn_model2(dummy_magnitude_spec, dummy_complex_spec_input, dummy_original_length)
    print(f"SpeqGCNNet2 Input magnitude shape: {dummy_magnitude_spec.shape}")
    print(f"SpeqGCNNet2 Input complex_spec shape: {dummy_complex_spec_input.shape}")
    print(f"SpeqGCNNet2 Output waveform shape: {output_waveform_gcn2.shape}")
    assert output_waveform_gcn2.ndim == 2 and output_waveform_gcn2.size(0) == batch, "Output shape mismatch for SpeqGCNNet2"

    print("\n--- SpeqGATNet2 Model Summary ---")
    speq_gat_model2 = SpeqGATNet2(n_channels=num_input_channels,
                                  n_classes=num_mask_classes,
                                  num_node=8,
                                  gat_heads=4,
                                  gat_dropout=0.6,
                                  n_fft=n_fft_val,
                                  hop_length=hop_length_val).to(device)
    print_model_summary(speq_gat_model2, batch, num_input_channels, num_freq_bins, num_time_frames)
    # Test forward pass
    output_waveform_gat2 = speq_gat_model2(dummy_magnitude_spec, dummy_complex_spec_input, dummy_original_length)
    print(f"SpeqGATNet2 Input magnitude shape: {dummy_magnitude_spec.shape}")
    print(f"SpeqGATNet2 Input complex_spec shape: {dummy_complex_spec_input.shape}")
    print(f"SpeqGATNet2 Output waveform shape: {output_waveform_gat2.shape}")
    assert output_waveform_gat2.ndim == 2 and output_waveform_gat2.size(0) == batch, "Output shape mismatch for SpeqGATNet2"


    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage (after initializations):")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")