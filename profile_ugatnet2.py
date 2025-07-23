import torch
from torch.profiler import profile, record_function, ProfilerActivity

# プロファイル対象のモデルをインポートします
from models.GNN import (
    UGAT2,
)  # models.urelnet から URelNet をインポートする代わりに、GCNからUGATNet2をインポート


def profile_model(model, device="cpu", batch_size=1, num_mic=1, length=128000):
    # モデルの初期化
    model.eval()  # 評価モードに設定

    # サンプル入力データの作成
    # 指定されたパラメータで入力データを作成
    x = torch.randn(batch_size, num_mic, length).to(device)

    # プロファイリングの実行
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,  # CUDAデバイスのプロファイリング (MPSの場合はProfilerActivity.MPS)
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./log/profile_ugatnet2"
        ),  # ログの出力先を変更
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # モデルの実行
        with record_function("model_inference"):
            output = model(x)
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
        )
    )  # 出力サイズを確認
    # プロファイリング結果の表示 (手動)
    print("\n=== プロファイリング結果 (UGAT2) ===")  # モデル名を明記
    key_avg_by_cuda_time = sorted(
        prof.key_averages(), key=lambda x: x.self_cuda_time_total, reverse=True
    )
    print(
        f"{'Event Name':<50} {'Self CUDA Time (Total)':<25} {'CUDA Time (Total)':<20} {'Count':<10}"
    )
    print("-" * 115)
    for event_avg in key_avg_by_cuda_time[:15]:  # 表示件数を少し増やす
        print(
            f"{event_avg.key:<50} {str(event_avg.self_cuda_time_total_str):<25} {str(event_avg.cuda_time_total_str):<20} {event_avg.count:<10}"
        )

    # メモリ使用量の表示 (手動)
    print("\n=== メモリ使用量 (UGAT2) ===")  # モデル名を明記
    key_avg_by_cuda_mem = sorted(
        prof.key_averages(), key=lambda x: x.self_cuda_memory_usage, reverse=True
    )
    print(
        f"{'Event Name':<50} {'Self CUDA Mem Usage':<25} {'CUDA Mem Usage':<20} {'Count':<10}"
    )
    print("-" * 115)
    for event_avg in key_avg_by_cuda_mem[:15]:  # 表示件数を少し増やす
        cuda_mem_usage_str = (
            f"{event_avg.cuda_memory_usage / (1024 * 1024):.2f} MB"
            if event_avg.cuda_memory_usage != 0
            else "0 B"
        )
        self_cuda_mem_usage_str = (
            f"{event_avg.self_cuda_memory_usage / (1024 * 1024):.2f} MB"
            if event_avg.self_cuda_memory_usage != 0
            else "0 B"
        )
        print(
            f"{event_avg.key:<50} {self_cuda_mem_usage_str:<25} {cuda_mem_usage_str:<20} {event_avg.count:<10}"
        )

    # トレースの保存
    prof.export_chrome_trace("trace_ugatnet2.json")  # トレースファイル名を変更
    print(
        "\nプロファイル結果は ./log/profile_ugatnet2 (TensorBoard) および trace_ugatnet2.json (Chrome Trace) に保存されました。"
    )
    print(
        "TensorBoardで確認するには、ターミナルで `tensorboard --logdir=./log/profile_ugatnet2` を実行してください。"
    )
    print(
        "Chrome Traceで確認するには、Chromeブラウザで chrome://tracing を開き、trace_ugatnet2.jsonをロードしてください。"
    )


if __name__ == "__main__":
    # デバイスの設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available(): # MPSのチェックも入れる場合
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    batch_size_main = 1
    num_mic_main = 1
    length_main = 128000  # 音声の長さ（サンプル数）

    # プロファイル対象のモデルをインスタンス化
    # UGAT2 の初期化パラメータを適切に設定してください
    model_to_profile = UGAT2(
        n_channels=num_mic_main,
        n_classes=1,  # マスク出力なので通常1
        hidden_dim=32,
        k_neighbors=8,
        gat_heads=4,
        gat_dropout=0.6,
    ).to(device)

    print(f"プロファイル対象モデル: {model_to_profile.__class__.__name__}")

    profile_model(
        model=model_to_profile,
        device=device,
        batch_size=batch_size_main,
        num_mic=num_mic_main,
        length=length_main,
    )
