import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from models.GCN import UGCNNet, UGCNNet2
# from models.SpeqGNN import SpeqGCNNet, SpeqGCNNet2 # 未使用のためコメントアウト
import time
import torch.optim as optim
from pathlib import Path

def profile_model(model, model_name, device="cpu", batch_size=1, num_mic=1, length=128000, output_dir="./log"):
    # モデルの初期化
    model.eval()  # 評価モードに設定

    # サンプル入力データの作成
    x = torch.randn(batch_size, num_mic, length).to(device)
    # yは学習ループ内で使われますが、プロファイル対象はモデルの推論とバックワードなので、ダミーデータで十分です
    y = torch.randn_like(x).to(device)

    # プロファイリングの実行
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss().to(device)

    # 出力ディレクトリの準備
    log_dir = Path(output_dir) / f"profile_{model_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA,],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(5): # wait+warmup+activeの合計ステップ数
            optimizer.zero_grad()
            with record_function("model_inference"):
                e = model(x)
            loss = loss_function(e, y)
            with record_function("model_backward"):
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()
            prof.step() # スケジューラを使う場合はステップを明示的に進める

    # --- プロファイル結果をコンソールに出力 ---
    print(f"\n--- Profiling Results for {model_name} ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    # --- プロファイル結果をマークダウンファイルに書き出し ---
    md_content = []
    md_content.append(f"# Profiling Report for {model_name}\n")

    # CUDA Time
    md_content.append("## Operator Performance (CUDA Time)\n")
    md_content.append("| Operator | Self CUDA Time | CUDA Time | Calls |\n")
    md_content.append("|---|---|---|---|\n")
    key_avg_by_cuda_time = sorted(prof.key_averages(), key=lambda k: k.self_cuda_time_total, reverse=True)
    for event in key_avg_by_cuda_time[:20]: # 上位20件
        if event.self_cuda_time_total > 0:
            md_content.append(f"| `{event.key}` | {event.self_cuda_time_total_str} | {event.cuda_time_total_str} | {event.count} |")
    md_content.append("\n")

    # Memory Usage
    md_content.append("## Memory Usage (CUDA)\n")
    md_content.append("| Operator | Self CUDA Mem | CUDA Mem | Calls |\n")
    md_content.append("|---|---|---|---|\n")
    key_avg_by_cuda_mem = sorted(prof.key_averages(), key=lambda k: k.self_cuda_memory_usage, reverse=True)
    for event in key_avg_by_cuda_mem[:20]: # 上位20件
        if event.self_cuda_memory_usage > 0:
            self_mem_str = f"{event.self_cuda_memory_usage / 1024**2:.2f} MB"
            total_mem_str = f"{event.cuda_memory_usage / 1024**2:.2f} MB"
            md_content.append(f"| `{event.key}` | {self_mem_str} | {total_mem_str} | {event.count} |")
    md_content.append("\n")

    # ファイルに書き込み
    md_filepath = log_dir / "profile_summary.md"
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print(f"プロファイル結果のサマリーを {md_filepath} に保存しました。")
    print(f"詳細なトレースは TensorBoard で確認できます: `tensorboard --logdir={log_dir}`")


if __name__ == '__main__':
    # デバイスの設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # MPSのチェックも入れる場合
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    batch_size_main = 1
    num_mic_main = 1
    length_main = 128000
    num_node_main = 8  # ノード数の設定

    model_list = ["UGCN", "UGCN2"]

    for model_type in model_list:
        print(f"\n===== Profiling {model_type} =====")
        if model_type == "UGCN":
            model = UGCNNet(n_channels=num_mic_main, n_classes=1, num_node=num_node_main).to(device)
        elif model_type == "UGCN2":
            model = UGCNNet2(n_channels=num_mic_main, n_classes=1, num_node=num_node_main).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        profile_model(
            model=model,
            model_name=model_type,
            device=device,
            batch_size=batch_size_main,
            num_mic=num_mic_main,
            length=length_main
        )
