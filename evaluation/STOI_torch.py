# coding:utf-8

import mymodule.const
from tqdm import tqdm
from mymodule import my_func, const
import torch  # torchをインポート
import torchaudio  # torchaudioをインポート
import torchmetrics  # torchmetricsをインポート

from mymodule.confirmation_GPU import get_device  # デバイス確認のための関数をインポート


def stoi_evaluation(target_data: torch.Tensor, estimation_data: torch.Tensor):
    """stoi値の算出 (torchmetricsを使用)

    :param target_data: 正解データのPyTorchテンソル
    :param estimation_data: モデル適用後データのPyTorchテンソル
    :return stoi_score: stoi値
    """

    device = get_device()  # 使用可能なデバイスを取得
    fs = mymodule.const.SR  # サンプリングレートを取得

    # torchmetricsのSTOIメトリックをインスタンス化
    # STOIも通常、CPUで計算されるため、デバイスはCPUに限定します。
    metric = torchmetrics.audio.STOI(fs=fs, extended=False).to(device)

    # 入力テンソルが正しい型とデバイスにあることを確認
    target_data = target_data.to(torch.float32).to(device)
    estimation_data = estimation_data.to(torch.float32).to(device)

    # NaNやInf値を0.0に置き換え
    target_data = torch.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
    estimation_data = torch.nan_to_num(estimation_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 長さを最短に合わせる (torchmetricsのSTOIも異なる長さの入力を処理できないため)
    min_length = min(target_data.shape[-1], estimation_data.shape[-1])
    target_data = target_data[..., :min_length]
    estimation_data = estimation_data[..., :min_length]

    # STOIスコアを計算 (torchmetricsは(batch, time)または(time)を期待)
    # 単一の音声ファイルの場合、形状を (1, num_samples) に変更します。
    if target_data.ndim == 1:
        target_data = target_data.unsqueeze(0)
    if estimation_data.ndim == 1:
        estimation_data = estimation_data.unsqueeze(0)

    stoi_score = metric(preds=estimation_data, target=target_data)

    return stoi_score.item()  # スコアをPythonのfloatとして返す


def stoi_main(target_dir, estimation_dir, out_path):
    """file list 取得"""
    target_list = my_func.get_file_list(dir_path=target_dir, ext=".wav")
    estimation_list = my_func.get_file_list(dir_path=estimation_dir, ext=".wav")

    """ 出力用のディレクトリーがない場合は作成 """
    my_func.make_dir(out_path)
    with open(out_path, "w") as file:
        file.write("target_name,estimation_name,stoi_score\n")

    for target_file, estimation_file in tqdm(zip(target_list, estimation_list)):
        """ファイル名の取得"""
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)  # get_file_listをget_file_nameに修正

        """ データの読み込み (torchaudioを使用) """
        target_data, fs_target = torchaudio.load(target_file)
        estimation_data, fs_estimation = torchaudio.load(estimation_file)

        # 複数チャネルの場合、最初のチャネルを選択（評価のため）
        if target_data.ndim > 1:
            target_data = target_data[0, :]
        if estimation_data.ndim > 1:
            estimation_data = estimation_data[0, :]

        # stoi_evaluation内で長さの調整とNaN/Inf処理が行われるため、ここでは行わない
        # stoi_evaluationに直接テンソルを渡す
        stoi_score = stoi_evaluation(target_data, estimation_data)

        with open(out_path, "a") as out_file:
            text = f"{target_name},{estimation_name},{stoi_score}\n"
            out_file.write(text)


if __name__ == "__main__":
    stoi_main(
        "../../sound_data/ConvtasNet/test/0dB/target",
        "../../sound_data/ConvtasNet/test/0dB/test",
        "../../sound_data/ConvtasNet/evaluation/stoi_CMU_0dB_before_spectorogram.csv",
    )

    stoi_main(
        "../../sound_data/ConvtasNet/test/0dB/target",
        "../../sound_data/ConvtasNet/result/0dB/test",
        "../../sound_data/ConvtasNet/evaluation/stoi_CMU_0dB_after_spectorogram.csv",
    )
