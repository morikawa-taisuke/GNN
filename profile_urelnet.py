import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from models.urelnet import URelNet
import time

def profile_model(model, device="cpu"):
    # モデルの初期化
    model.eval()  # 評価モードに設定

    # サンプル入力データの作成
    x = torch.randn(batch_size, num_mic, length).to(device)

    # プロファイリングの実行
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.MPS,  # MPSデバイスのプロファイリング
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # モデルの実行
        with record_function("model_inference"):
            output = model(x)

    # プロファイリング結果の表示
    print("\n=== プロファイリング結果 ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))
    
    # メモリ使用量の表示
    print("\n=== メモリ使用量 ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage", row_limit=10))

    # トレースの保存
    prof.export_chrome_trace("trace.json")

if __name__ == '__main__':
    # デバイスの設定
    device = "mps"
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    batch_size = 1
    num_mic = 1
    length = 128000

    model = URelNet(n_channels=num_mic, n_classes=num_mic, k_neighbors=8).to(device)

    profile_model(model=model, device=device) 
