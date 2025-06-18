import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from models.urelnet import URelNet
from models.GCN import UGATNet2
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
            ProfilerActivity.CUDA,  # MPSデバイスのプロファイリング
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

    # プロファイリング結果の表示 (手動)
    print("\n=== プロファイリング結果 ===")
    key_avg_by_cuda_time = sorted(prof.key_averages(), key=lambda x: x.self_cuda_time_total, reverse=True)
    print(f"{'Event Name':<50} {'Self CUDA Time (Total)':<25} {'CUDA Time (Total)':<20} {'Count':<10}")
    print("-" * 115)
    for event_avg in key_avg_by_cuda_time[:10]:
        print(f"{event_avg.key:<50} {str(event_avg.self_cuda_time_total_str):<25} {str(event_avg.cuda_time_total_str):<20} {event_avg.count:<10}")

    # メモリ使用量の表示 (手動)
    print("\n=== メモリ使用量 ===")
    key_avg_by_cuda_mem = sorted(prof.key_averages(), key=lambda x: x.self_cuda_memory_usage, reverse=True)
    print(f"{'Event Name':<50} {'Self CUDA Mem Usage':<25} {'CUDA Mem Usage':<20} {'Count':<10}")
    print("-" * 115)
    for event_avg in key_avg_by_cuda_mem[:10]:
        cuda_mem_usage_str = f"{event_avg.cuda_memory_usage / (1024 * 1024):.2f} MB" if event_avg.cuda_memory_usage !=0 else "0 B"
        self_cuda_mem_usage_str = f"{event_avg.self_cuda_memory_usage / (1024 * 1024):.2f} MB" if event_avg.self_cuda_memory_usage !=0 else "0 B"
        print(f"{event_avg.key:<50} {self_cuda_mem_usage_str:<25} {cuda_mem_usage_str:<20} {event_avg.count:<10}")

    # トレースの保存
    prof.export_chrome_trace("trace.json")
    print("\nプロファイル結果は ./log/profile (TensorBoard) および trace.json (Chrome Trace) に保存されました。")
    print("TensorBoardで確認するには、ターミナルで `tensorboard --logdir=./log/profile` を実行してください。")
    print("Chrome Traceで確認するには、Chromeブラウザで chrome://tracing を開き、trace.jsonをロードしてください。")


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

    model = UGATNet2(n_channels=num_mic, n_classes=1, k_neighbors=8, gat_heads=4, gat_dropout=0.6).to(device)

    profile_model(model=model, device=device) 
