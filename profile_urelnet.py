import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from models.GCN import UGCNNet, UGCNNet2
from models.SpeqGNN import SpeqGCNNet, SpeqGCNNet2
import time
import torch.optim as optim

def profile_model(model, device="cpu"):
    # モデルの初期化
    model.eval()  # 評価モードに設定

    # サンプル入力データの作成
    y = torch.randn(batch_size, num_mic, length).to(device)
    x = torch.randn(batch_size, num_mic, length).to(device)

    # プロファイリングの実行
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizerを選択(Adam)
    loss_function = nn.MSELoss().to(device) # 損失関数に使用する式の指定(最小二乗誤差)
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        tortal_loss = 0.0
        optimizer.zero_grad()  # 勾配の初期化
        for _ in enumerate(range(10)):
            x, y = x.to(device), y.to(device)
            with record_function("inference"):
                e = model(x)
            loss = loss_function(e, y)
            with record_function("backward"):
                loss.backward()
            with record_function("step"):
                optimizer.step()
            tortal_loss += loss.detach()

    print(prof.key_averages().table(sort_by = "cuda_time", row_limit = 10))


if __name__ == '__main__':
    # デバイスの設定
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    batch_size = 1
    num_mic = 1
    length = 128000
    num_node = 8  # ノード数の設定

    model_list = ["UGCN", "UGCN2"]

    for model_type in model_list:
        if model_type == "UGCN":
            model = UGCNNet(n_channels=num_mic, n_classes=1, num_node=num_node).to(device)
        elif model_type == "UGCN2":
            model = UGCNNet2(n_channels=num_mic, n_classes=1, num_node=num_node).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        profile_model(model=model, device=device) 
