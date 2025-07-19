import os
import sys
import torch
import torchaudio
import torchmetrics  # torchmetricsをインポート

# mymoduleのパスを適切に設定してください（必要に応じてコメントを解除し、パスを修正してください）
# sys.path.append('C:\\Users\\kataoka-lab\\Desktop\\hikitugi_conv\\ConvTasNet\\mymodule\\')
from mymodule import my_func
from mymodule.confirmation_GPU import get_device

from tqdm import tqdm


def pesq_evaluation(target_data: torch.Tensor, estimation_data: torch.Tensor):
    """pesq値の算出 (torchmetricsを使用)

    :param target_data: 正解データのPyTorchテンソル
    :param estimation_data: モデル適用後データのPyTorchテンソル
    :return pesq_score: pesq値
    """
    device = get_device()  # 使用可能なデバイスを取得
    # torchmetricsのPESQメトリックをインスタンス化
    # fs=16000, mode='wb' (Wideband) を仮定。必要に応じて調整してください。
    # PESQは通常、CPUで計算されるため、デバイスはCPUに限定します。
    metric = torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality(fs=16000, mode="wb").to(device)

    # 入力テンソルが正しい型とデバイスにあることを確認
    target_data = target_data.to(torch.float32).to(device)
    estimation_data = estimation_data.to(torch.float32).to(device)

    # torchmetricsのPESQは、(batch_size, num_samples) または (num_samples,) 形式を期待します。
    # 入力が単一の音声ファイルである場合、形状を (1, num_samples) に変更します。
    if target_data.ndim == 1:
        target_data = target_data.unsqueeze(0)
    if estimation_data.ndim == 1:
        estimation_data = estimation_data.unsqueeze(0)

    # NaN/Inf値を0.0に置き換え
    target_data = torch.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
    estimation_data = torch.nan_to_num(estimation_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 長さを最短に合わせる (torchmetricsのPESQは異なる長さの入力を処理できないため)
    min_length = min(target_data.shape[-1], estimation_data.shape[-1])
    target_data = target_data[..., :min_length]
    estimation_data = estimation_data[..., :min_length]

    # メトリックを更新して計算
    pesq_score = metric(preds=estimation_data, target=target_data)

    return pesq_score.item()  # スコアをPythonのfloatとして返す


def pesq_main(target_dir, estimation_dir, out_path):
    """pesqの算出とcsvファイルへの書き込み

    :param target_dir: 正解データのディレクトリ
    :param estimation_dir: モデル適用後データのディレクトリ
    :param dataset_dir:
    """
    """ 出力ファイルの作成"""
    my_func.make_dir(out_path)
    with open(out_path, "w") as out_file:
        out_file.write("target_name,estimation_name,pesq_score\n")

    """ ファイルリストの作成 """
    target_list = my_func.get_file_list(dir_path=target_dir, ext=".wav")
    estimation_list = my_func.get_file_list(dir_path=estimation_dir, ext=".wav")

    for target_file, estimation_file in tqdm(zip(target_list, estimation_list)):
        """ファイル名の取得"""
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)

        """ 音源の読み込み (torchaudioを使用し、テンソルとして扱う) """
        target_data, sr_target = torchaudio.load(target_file)
        estimation_data, sr_estimation = torchaudio.load(estimation_file)

        # 複数チャネルの場合、最初のチャネルを選択（評価のため）
        if target_data.ndim > 1:
            target_data = target_data[0, :]
        if estimation_data.ndim > 1:
            estimation_data = estimation_data[0, :]

        # PESQは異なるサンプリングレートを処理できないため、一致していることを確認するか、リサンプリングが必要
        # この例では、16kHzを想定しています。
        # 必要に応じて、torch_audios.functional.resampleなどを使用してリサンプリングロジックを追加してください。

        """ pesq値の計算 """
        # pesq_evaluation内でテンソルの形状と長さの調整、NaN/Inf処理、デバイス変換が行われます。
        pesq_score = pesq_evaluation(target_data, estimation_data)

        """ 出力(ファイルへの書き込み) """
        with open(out_path, "a") as out_file:
            text = f"{target_name},{estimation_name},{pesq_score}\n"
            out_file.write(text)


if __name__ == "__main__":
    target_dir = "../../sound_data/LSTM/mix_data/JA_hoth_10db_05sec/test"  # 正解データ
    estimation_dir = "../../sound_data/UNet/result/JA_hoth_10db_05sec"  # モデルデータ
    out_name = "JA_hoth_10db_05sec"

    pesq_main(target_dir=target_dir, estimation_dir=estimation_dir, out_path=f"pesq1/{out_name}.csv")
