import torch
from tqdm import tqdm
from mymodule import my_func
import torchaudio  # torchaudioをインポート
import torchmetrics  # torchmetricsをインポート

from mymodule.confirmation_GPU import get_device


def sisdr_evaluation(target_data: torch.Tensor, estimation_data: torch.Tensor, eps: float = 1e-8):
    """SI-SDRを算出 (torchmetricsを使用)

    :param target_data: 正解データのPyTorchテンソル
    :param estimation_data: モデル適用後データのPyTorchテンソル
    :param eps: ゼロ除算を避けるための小さな値
    :return sisdr_score: sisdr値
    """
    device = get_device()  # 使用可能なデバイスを取得
    # torchmetricsのSI-SDRメトリックをインスタンス化
    # SI-SDRは通常、CPUで計算されるため、デバイスはCPUに限定します。
    metric = torchmetrics.audio.SI_SDR().to(device)

    # 入力テンソルが正しい型とデバイスにあることを確認
    target_data = target_data.to(torch.float32).to(device)
    estimation_data = estimation_data.to(torch.float32).to(device)

    # NaN/Inf値を0.0に置き換え
    target_data = torch.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
    estimation_data = torch.nan_to_num(estimation_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 長さを最短に合わせる (torchmetricsのSI-SDRも異なる長さの入力を処理できないため)
    # torchmetricsは自動で調整することもありますが、明示的に行うことで一貫性を保ちます。
    min_length = min(target_data.shape[-1], estimation_data.shape[-1])
    target_data = target_data[..., :min_length]
    estimation_data = estimation_data[..., :min_length]

    # SI-SDRスコアを計算 (torchmetricsは(batch, time)または(time)を期待)
    # 単一の音声ファイルの場合、形状を (1, num_samples) に変更します。
    if target_data.ndim == 1:
        target_data = target_data.unsqueeze(0)
    if estimation_data.ndim == 1:
        estimation_data = estimation_data.unsqueeze(0)

    sisdr_score = metric(preds=estimation_data, target=target_data)

    return sisdr_score.item()  # スコアをPythonのfloatとして返す


def sisdr_main(target_dir, estimation_dir, out_path):
    """sisdrを計算する

    :param target_dir: 正解データのディレクトリ
    :param estimation_dir: モデル適用後データのディレクトリ
    :return sisdr_score: sisdr値
    """

    """ file list 取得 """
    target_list = my_func.get_file_list(dir_path=target_dir, ext=".wav")
    estimation_list = my_func.get_file_list(dir_path=estimation_dir, ext=".wav")

    """ 出力用のディレクトリーがない場合は作成 """
    my_func.make_dir(out_path)
    """ ファイルの書き込み """
    with open(out_path, "w") as out_file:
        out_file.write("target_name,estimation_name,SI-SDR_score\n")
    sisdr_sum = 0

    for target_file, estimation_file in tqdm(zip(target_list, estimation_list)):
        """ファイル名の取得"""
        target_name, _ = my_func.get_file_name(target_file)
        estimation_name, _ = my_func.get_file_name(estimation_file)

        """ データの読み込み (torchaudioを使用) """
        target_data, fs_target = torchaudio.load(target_file)
        estimation_data, fs_estimation = torchaudio.load(estimation_file)

        # 複数チャネルの場合、最初のチャネルを選択（評価のため）
        if target_data.ndim > 1:
            target_data = target_data[0, :]
        if estimation_data.ndim > 1:
            estimation_data = estimation_data[0, :]

        # sisdr_evaluation内で長さの調整とNaN/Inf処理が行われるため、ここでは行わない
        # sisdr_evaluationに直接テンソルを渡す
        sisdr_score = sisdr_evaluation(target_data=target_data, estimation_data=estimation_data)

        sisdr_sum += sisdr_score

        """ スコアの書き込み """
        with open(out_path, "a") as out_file:
            text = f"{target_name},{estimation_name},{sisdr_score}\n"
            out_file.write(text)
    """average"""
    sisdr_ave = sisdr_sum / len(estimation_list)
    with open(out_path, "a") as out_file:
        text = f"average,,{sisdr_ave}\n"
        out_file.write(text)


if __name__ == "__main__":
    noise_list = ["white", "hoth"]
    learning_list = ["noise_only_delay", "reverbe_only_delay", "noise_reverbe_delay"]
    for noise in noise_list:
        sisdr_main(
            target_dir=f"../../../sound_data/mix_data/02_08/JA_{noise}_10dB_07sec/test/clean",
            estimation_dir=f"../../../sound_data/mix_data/02_08/JA_{noise}_10dB_07sec/test/clean",
            out_path=f"./02_08/JA_{noise}_10dB_07sec_clean_data.csv",
        )
