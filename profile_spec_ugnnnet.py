import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from models.SpeqGNN import SpeqGCNNet, SpeqGCNNet2
# from models.GCN import UGCNNet, UGCNNet2 # 未使用のためコメントアウト
import time
import torch.optim as optim
from pathlib import Path

def profile_model(model, model_name, device="cpu", batch_size=1, num_mic=1, length=128000, output_dir="./log"):
    # モデルの初期化
    model.eval()  # 評価モードに設定

    # --- STFTパラメータ (モデルの設計に合わせて調整してください) ---
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.hann_window(win_length, device=device)

    # サンプル入力データの作成
    x_time = torch.randn(batch_size, num_mic, length, device=device)

    # --- 入力データをSTFTでスペクトログラムに変換 ---
    x_magnitude_spec = torch.stft(x_time.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=False)
    x_magnitude_spec = torch.sqrt(x_magnitude_spec[..., 0]**2 + x_magnitude_spec[..., 1]**2).unsqueeze(1) # (B, 1, F, T_spec)

    # 複素スペクトログラム (B, F, T_spec)
    x_complex_spec = torch.stft(x_time.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

    original_len = x_time.shape[-1]

    # y (教師データ) も同じ形状のダミーデータを作成します
    y = torch.randn_like(x_time).to(device)

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
                e = model(x_magnitude_spec, x_complex_spec, original_len)
            loss = loss_function(e, y)
            with record_function("model_backward"):
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()
            prof.step() # スケジューラを使う場合はステップを明示的に進める

    # --- プロファイル結果をコンソールに出力 ---
    print(f"\n--- Profiling Results for {model_name} ---")
    # デバイスに応じてソートキーを変更
    sort_key = "self_cuda_time_total" if "cuda" in device.type else "self_cpu_time_total"
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

    model_list = ["SpeqGCN", "SpeqGCN2"]

    for model_type in model_list:
        print(f"\n===== Profiling {model_type} =====")
        if model_type == "SpeqGCN":
            model = SpeqGCNNet(n_channels=num_mic_main, n_classes=1, num_node=num_node_main).to(device)
        elif model_type == "SpeqGCN2":
            model = SpeqGCNNet2(n_channels=num_mic_main, n_classes=1, num_node=num_node_main).to(device)
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
