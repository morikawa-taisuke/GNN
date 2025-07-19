import torch


# 適切なデバイスを選択する関数
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    print("torch version:", torch.__version__)  # torch version:1.4.0 torchのバージョン
    device = get_device()  # 使用可能なデバイスを取得
    print("使用可能なデバイス:", device)  # 使用可能なデバイスを表示

    if device.type == "cuda":  # 使用可能なGPUがcudaである場合
        # CUDAの設定を表示
        print("CUDA is available.")
        print("Number of GPUs available in PyTorch:", torch.cuda.device_count())  # 使えるGPUの個数?
        print("Current CUDA device:", torch.cuda.current_device())  # 現在のCUDAデバイス
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))  # 現在のCUDAデバイスの名前
        print("CUDA device version:", torch.cuda.get_device_capability(torch.cuda.current_device()))  # CUDAデバイスのバージョン
    elif device.type == "mps":  # 使用可能なGPUがMPSである場合
        print("MPS is available.")
    else:
        print("No GPU available, using CPU.")
