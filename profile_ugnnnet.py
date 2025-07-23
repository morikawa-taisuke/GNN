from pathlib import Path

import torch
import torch.nn as nn
# from models.SpeqGNN import SpeqGCNNet, SpeqGCNNet2 # 未使用のためコメントアウト
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

from models.GNN import UGCN, UGCN2


def profile_model(
    model,
    model_name,
    device="cpu",
    batch_size=1,
    num_mic=1,
    length=128000,
    output_dir="./log",
):
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
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):  # wait+warmup+activeの合計ステップ数
            optimizer.zero_grad()
            with record_function("model_inference"):
                e = model(x)
            loss = loss_function(e, y)
            with record_function("model_backward"):
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()
            prof.step()  # スケジューラを使う場合はステップを明示的に進める

    # --- プロファイル結果をコンソールに出力 ---
    print(f"\n--- Profiling Results for {model_name} ---")
    # デバイスに応じてソートキーを変更
    sort_key = (
        "self_cuda_time_total" if "cuda" in device.type else "self_cpu_time_total"
    )
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    # --- プロファイル結果をテキストファイルに書き出し ---
    # コンソールに出力したテーブルと同じ内容を文字列として取得
    # row_limitを少し多めにして、より詳細な情報をファイルに残します
    profile_summary_text = prof.key_averages().table(sort_by=sort_key, row_limit=30)

    # ファイルに書き込み
    txt_filepath = log_dir / "profile_summary.txt"
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(f"--- Profiling Results for {model_name} ---\n")
        f.write(f"Device: {device.type}\n\n")
        f.write(profile_summary_text)

    print(f"プロファイル結果のサマリーを {txt_filepath} に保存しました。")
    print(
        f"詳細なトレースは TensorBoard で確認できます: `tensorboard --logdir={log_dir}`"
    )


if __name__ == "__main__":
    # デバイスの設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # MPSのチェックも入れる場合
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    batch_size_main = 1
    num_mic_main = 1
    length_main = 16000 * 8  # 8秒の音声データ (例)
    num_node_main = 8  # ノード数の設定

    model_list = ["UGCN", "UGCN2"]

    for model_type in model_list:
        print(f"\n===== Profiling {model_type} =====")
        if model_type == "UGCN":
            model = UGCN(
                n_channels=num_mic_main, n_classes=1, num_node=num_node_main
            ).to(device)
        elif model_type == "UGCN2":
            model = UGCN2(
                n_channels=num_mic_main, n_classes=1, num_node=num_node_main
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        profile_model(
            model=model,
            model_name=model_type,
            device=device,
            batch_size=batch_size_main,
            num_mic=num_mic_main,
            length=length_main,
        )
