import torch
import matplotlib.pyplot as plt

def overlap_save(input, flame_size):
    """    オーバーラップアドを用いて入力信号とフィルタの畳み込みを計算する関数

    Args:
        input (_type_): _description_
        flame_size (_type_): _description_
    """
    print("overlap_save: 開始")

    hop_size = flame_size // 2  # ホップサイズはフィルタ長の半分
    input_padded = torch.cat((torch.zeros(hop_size), input))    # 入力信号の前に0を追加

    out = torch.zeros(len(input_padded))   # 出力用の配列

    num_flame = len(input_padded) // hop_size  # フレーム数の計算

    for i in range(num_flame):
        start = i * hop_size
        end = start + flame_size
        
        # ブロックを取得
        flame = input_padded[start:end]

        # 窓かけ
        window = torch.hann_window(len(flame))
        flame_windowed = flame * window  # ハニング窓を適用

        # 出力に結果を加算
        end = min(len(out), end)  # 出力の長さを調整
        out[start:end] += flame_windowed.clone().detach()
    return out[hop_size:]  # ホップサイズ分の先頭を削除して返す


# --- 実行例 ---
if __name__ == '__main__':
    # パラメータ設定
    SAMPLE_RATE = 16000  # サンプリングレート
    SIGNAL_LENGTH = SAMPLE_RATE * 8  # 信号の長さ（2秒）
    FLAME_SIZE = int(SAMPLE_RATE * 0.1)  # フレームサイズ（100ms）

    HOP_SIZE = FLAME_SIZE // 2  # ホップサイズ（50%オーバーラップ）
    
    # ランダムな入力信号とフィルタを生成
    x = torch.randn(SIGNAL_LENGTH)

    print("--- パラメータ ---")
    print(f"入力信号の長さ (len(x)): {x.numel()}")
    print(f"フレームサイズ (FLAME_SIZE): {FLAME_SIZE}")
    print(f"ホップサイズ (HOP_SIZE = FLAME_SIZE/2): {HOP_SIZE}  <-- 50%のオーバーラップ\n")
    
    # オーバーラップセーブ法で畳み込みを計算
    y_os = overlap_save(x, flame_size=FLAME_SIZE)

    # --- 結果の検証 ---
    print("--- 結果の検証 ---")

    if y_os.numel() < x.numel():
        x = x[:y_os.numel()]  # オーバーラップセーブ法の結果に合わせて切り詰める
    elif y_os.numel() > x.numel():
        y_os = y_os[:x.numel()]
    is_close = torch.allclose(x, y_os, atol=1e-5)
    print(f"両者の結果は一致するか？ -> {is_close}")
    print(f"input")
    print(x.tolist())
    print(f"overlap")
    print(y_os.tolist())
    is_close = torch.mean(x-y_os)
    print(f"両者の差の合計: {is_close}")

    # --- 結果のプロット ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    plt.title("オーバーラップセーブ法 と np.convolve の結果比較", fontsize=14, fontname='MS Gothic')
    plt.plot(x[:500], label='origin', linestyle='-', color='red', linewidth=1.5)
    plt.plot(y_os[:500], label='Overlap-Save (50% Overlap)', linestyle='--', color='green', linewidth=1.5)
    plt.xlabel("サンプル", fontname='MS Gothic')
    plt.ylabel("振幅", fontname='MS Gothic')
    plt.legend(prop={"family":"MS Gothic"})
    plt.grid(True)
    plt.show()