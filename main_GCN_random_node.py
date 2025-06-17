from models.wave_unet import U_Net
from models.GCN import UGCNNet
import time             # 時間
# from librosa.core import stft, istft
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.contrib import tenumerate
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from itertools import permutations
from torch.nn.utils import weight_norm
# import scipy.signal as sp
# import scipy as scipy
from torchinfo import summary
import os
from pathlib import Path

import UGNNNet_DatasetClass
from mymodule import my_func, const


# CUDAのメモリ管理設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDAの可用性をチェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("mps")
print(f"Using device: {device}")

def padding_tensor(tensor1, tensor2):
    """
    最後の次元（例: 時系列長）が異なる2つのテンソルに対して、
    短い方を末尾にゼロパディングして長さをそろえる。

    Args:
        tensor1, tensor2 (torch.Tensor): 任意の次元数のテンソル

    Returns:
        padded_tensor1, padded_tensor2 (torch.Tensor)
    """
    len1 = tensor1.size(-1)
    len2 = tensor2.size(-1)
    max_len = max(len1, len2)

    pad1 = [0, max_len - len1]  # 最後の次元だけパディング
    pad2 = [0, max_len - len2]

    padded_tensor1 = F.pad(tensor1, pad1)
    padded_tensor2 = F.pad(tensor2, pad2)

    return padded_tensor1, padded_tensor2

def sisdr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisdr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError("Dimention mismatch when calculate si-sdr, {} vs {}".format(x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm * s_zm, dim=-1,keepdim=True) * s_zm / torch.sum(s_zm * s_zm, dim=-1,keepdim=True)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(t - x_zm) + eps))

def si_sdr_loss(ests, egs):
    # spks x n x S
    # ests: estimation
    # egs: target
    refs = egs
    num_speeker = len(refs)
    #print("spks", num_speeker)
    # print(f"ests:{ests.shape}")
    # print(f"egs:{egs.shape}")

    def sisdr_loss(permute):
        # for one permute
        #print("permute", permute)
        return sum([sisdr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = egs.size(0)
    sisdr_mat = torch.stack([sisdr_loss(p) for p in permutations(range(num_speeker))])
    max_perutt, _ = torch.max(sisdr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N


def train(clean_path:str, noisy_path:str, out_path:str="./RESULT/pth/result.pth", loss_func:str="stft_MSE", batchsize:int=const.BATCHSIZE, checkpoint_path:str=None, train_count:int=const.EPOCH, earlystopping_threshold:int=5):
    """ GPUの設定 """
    device = "cuda" if torch.cuda.is_available() else "cpu" # GPUが使えれば使う
    """ その他の設定 """
    out_path = Path(out_path)   # path型に変換
    out_name = out_path.stem
    out_dir = out_path.parent
    writer = SummaryWriter(log_dir=f"{const.LOG_DIR}\\{out_name}")  # logの保存先の指定("tensorboard --logdir ./logs"で確認できる)
    now = my_func.get_now_time()
    csv_path = f"{const.LOG_DIR}\\{out_name}\\{out_name}_{now}.csv"
    my_func.make_dir(csv_path)
    with open(csv_path, "w") as csv_file:  # ファイルオープン
        csv_file.write(f"dataset,out_name,loss_func\n{noisy_path},{out_path},{loss_func}")

    """ Early_Stoppingの設定 """
    best_loss = np.inf  # 損失関数の最小化が目的の場合，初めのbest_lossを無限大にする
    earlystopping_count = 0

    """ Load dataset データセットの読み込み """
    # dataset = datasetClass.TasNet_dataset_csv(args.dataset, channel=channel, device=device) # データセットの読み込み
    dataset = UGNNNet_DatasetClass.AudioDataset(clean_path, noisy_path) # データセットの読み込み
    dataset_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)


    """ ネットワークの生成 """
    model = UGCNNet(n_channels=1, n_classes=1, k_neighbors=8).to(device)
    # model = U_Net().to(device)
    # print(f"\nmodel:{model}\n")                           # モデルのアーキテクチャの出力
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # optimizerを選択(Adam)
    if loss_func != "SISDR":
        loss_function = nn.MSELoss().to(device)                 # 損失関数に使用する式の指定(最小二乗誤差)

    """ チェックポイントの設定 """
    if checkpoint_path != None:
        print("restart_training")
        checkpoint = torch.load(checkpoint_path)  # checkpointの読み込み
        model.load_state_dict(checkpoint["model_state_dict"])  # 学習途中のモデルの読み込み
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # オプティマイザの読み込み
        # optimizerのstateを現在のdeviceに移す。これをしないと、保存前後でdeviceの不整合が起こる可能性がある。
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
    else:
        start_epoch = 1

    """ 学習の設定を出力 """
    print("====================")
    print("device: ", device)
    print("out_path: ", out_path)
    print("dataset: ", noisy_path)
    print("loss_func: ", loss_func)
    print("====================")

    my_func.make_dir(out_dir)
    model.train()                   # 学習モードに設定

    start_time = time.time()    # 時間を測定
    epoch = 0
    for epoch in range(start_epoch, train_count+1):   # 学習回数
        model_loss_sum = 0              # 総損失の初期化
        print("Train Epoch:", epoch)    # 学習回数の表示
        for _, (mix_data, target_data) in tenumerate(dataset_loader):
            """ モデルの読み込み """
            mix_data, target_data = mix_data.to(device), target_data.to(device) # データをGPUに移動

            """ 勾配のリセット """
            optimizer.zero_grad()  # optimizerの初期化

            """ データの整形 """
            mix_data = mix_data.to(torch.float32)   # target_dataのタイプを変換 int16→float32
            target_data = target_data.to(torch.float32) # target_dataのタイプを変換 int16→float32
            # mix_data = mix_data.unsqueeze(dim=0)    # [バッチサイズ, マイク数，音声長]
            # target_data = target_data[np.newaxis, :, :] # 次元を増やす[1,音声長]→[1,1,音声長]
            # print("mix:", mix_data.shape)

            """ モデルに通す(予測値の計算) """
            # print("model_input", mix_data.shape)
            estimate_data = model(mix_data) # モデルに通す

            """ データの整形 """
            # print("estimation:", estimate_data.shape)
            # print("target:", target_data.shape)
            estimate_data, target_data = padding_tensor(estimate_data, target_data)

            """ 損失の計算 """
            model_loss = 0
            match loss_func:
                case "SISDR":
                    model_loss = si_sdr_loss(estimate_data, target_data[0])
                case "wave_MSE":
                    model_loss = loss_function(estimate_data, target_data)  # 時間波形上でMSEによる損失関数の計算
                case "stft_MSE":
                    """ 周波数軸に変換 """
                    stft_estimate_data = torch.stft(estimate_data, n_fft=1024, return_complex=False)
                    stft_target_data = torch.stft(target_data[0], n_fft=1024, return_complex=False)
                    model_loss = loss_function(stft_estimate_data, stft_target_data)  # 時間周波数上MSEによる損失の計算

            model_loss_sum += model_loss  # 損失の加算

            """ 後処理 """
            model_loss.backward()           # 誤差逆伝搬
            optimizer.step()                # 勾配の更新

            del mix_data, target_data, estimate_data, model_loss    # 使用していない変数の削除
            torch.cuda.empty_cache()    # メモリの解放 1iterationごとに解放

        """ チェックポイントの作成 """
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": model_loss_sum},
                    f"{out_dir}/{out_name}_ckp.pth")

        writer.add_scalar(str(out_name[0]), model_loss_sum, epoch)
        #writer.add_scalar(str(str_name[0]) + "_" + str(a) + "_sisdr-sisnr", model_loss_sum, epoch)
        print(f"[{epoch}]model_loss_sum:{model_loss_sum}")  # 損失の出力

        torch.cuda.empty_cache()    # メモリの解放 1iterationごとに解放
        with open(csv_path, "a") as out_file:  # ファイルオープン
            out_file.write(f"{model_loss_sum}\n")  # 書き込み
        # torch.cuda.empty_cache()    # メモリの解放 1epochごとに解放-

        """ Early_Stopping の判断 """
        # best_lossとmodel_loss_sumを比較
        if model_loss_sum < best_loss:  # model_lossのほうが小さい場合
            print(f"{epoch:3} [epoch] | {model_loss_sum:.6} <- {best_loss:.6}")
            my_func.make_dir(out_dir)  # best_modelの保存
            torch.save(model.to(device).state_dict(), f"{out_dir}/BEST_{out_name}.pth")  # 出力ファイルの保存
            best_loss = model_loss_sum  # best_lossの変更
            earlystopping_count = 0
        else:
            earlystopping_count += 1
            if (epoch > 100) and (earlystopping_count > earlystopping_threshold):
                break
        if epoch == 100:
            torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")  # 出力ファイルの保存

    """ 学習モデル(pthファイル)の出力 """
    print("model save")
    my_func.make_dir(out_dir)
    torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")         # 出力ファイルの保存

    writer.close()

    """ 学習時間の計算 """
    time_end = time.time()              # 現在時間の取得
    time_sec = time_end - start_time    # 経過時間の計算(sec)
    time_h = float(time_sec)/3600.0     # sec->hour
    print(f"time：{str(time_h)}h")      # 出力

def test(mix_dir:str, out_dir:str, model_path:str):
    filelist_mixdown = my_func.get_file_list(mix_dir)
    print('number of mixdown file', len(filelist_mixdown))


    # ディレクトリを作成
    my_func.make_dir(out_dir)

    model_dir = my_func.get_dir_name(model_path)
    model_name, _ = my_func.get_file_name(model_path)

    # モデルの読み込み
    model = UGCNNet(n_channels=1, n_classes=1, k_neighbors=8).to(device)
    # model = U_Net().to(device)

    # TasNet_model.load_state_dict(torch.load('./pth/model/' + model_path + '.pth'))
    model.load_state_dict(torch.load(os.path.join(model_dir, f"BEST_{model_name}.pth")))
    # TCN_model.load_state_dict(torch.load('reverb_03_snr20_reverb1020_snr20-clean_DNN-WPE_TCN_100.pth'))

    for fmixdown in tqdm(filelist_mixdown):  # filelist_mixdownを全て確認して、それぞれをfmixdownに代入
        # mixは振幅、prmはパラメータ
        mix, prm = my_func.load_wav(fmixdown)  # waveでロード
        # print(f'mix.shape:{mix.shape}')
        mix = torch.from_numpy(mix)
        mix = mix.to(device)
        mix = mix.to(torch.float32)
        # mix = mix.unsqueeze(dim=0)    # [バッチサイズ, マイク数，音声長]
        # print("mix: ", mix.shape)
        mix = mix.unsqueeze(dim=0)    # [バッチサイズ, マイク数，音声長]
        # print("mix: ", mix.shape)
        mix_max = torch.max(mix)  # 最大値の取得
        # mix = my_func.load_audio(fmixdown)     # torchaoudioでロード

        mix = mix[np.newaxis, :]
        # print(f"mix:{type(mix)}")

        # print(f'mix.shape:{mix.shape}')  # mix.shape=[1,チャンネル数×音声長]
        mix = torch.tensor(mix, dtype=torch.float32)
        # mix = split_data(mix, channel=channels)  # mix=[チャンネル数,音声長]
        # print(f'mix.shape:{mix.shape}')
        # mix = mix[np.newaxis, :, :]  # mix=[1,チャンネル数,音声長]
        # mix = torch.from_numpy(mix)
        # print("00mix", mix.shape)
        mix = mix.to("cuda")
        # print("11mix", mix.shape)
        mix = mix / (mix.abs().max() + 1e-8)
        separate = model(mix)  # モデルの適用
        # print("separate", separate.shape)
        # separate = separate.cpu()
        # separate = separate.detach().numpy()
        # separate = separate[0, 0, :]
        # print(f'separate.shape:{separate.shape}')
        # mix_max=mix_max.detach().numpy()
        # separate_max=torch.max(separate)
        # separate_max=separate_max.detach().numpy()

        separate = separate * (mix_max / torch.max(separate))
        separate = separate.cpu()
        separate = separate.detach().numpy()


        # 分離した speechを出力ファイルとして保存する。
        # 拡張子を変更したパス文字列を作成
        foutname, _ = os.path.splitext(os.path.basename(fmixdown))
        # ファイル名とフォルダ名を結合してパス文字列を作成
        fname = os.path.join(out_dir, (foutname + '.wav'))
        # print('saving... ', fname)
        # 混合データを保存
        # mask = mask*mix
        my_func.save_wav(fname, separate, prm)
        torch.cuda.empty_cache()    # メモリの解放 1音声ごとに解放
        # torchaudio.save(
        #     fname,
        #     separate.detach().numpy(),
        #     const.SR,
        #     format='wav',
        #     encoding='PCM_S',
        #     bits_per_sample=16
        # )



if __name__ == '__main__':
    # "C:\Users\kataoka-lab\Desktop\sound_data\dataset\subset_DEMAND_hoth_1010dB_1ch\subset_DEMAND_hoth_1010dB_05sec_1ch\noise_reverbe"

    wave_type = "noise_only"
    # train(clean_path=f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/train/clean",
    #       noisy_path=f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/train/{wave_type}",
    #       out_path=f"{const.PTH_DIR}/UGCN/subset_DEMAND_1ch/random_node/{wave_type}2.pth", batchsize=1)

    # test(mix_dir=f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/test/{wave_type}",
    #      out_dir=f"{const.OUTPUT_WAV_DIR}/UGCN/subset_DEMAND_1ch/random_node/STFT_MSE/{wave_type}",
    #      model_path=f"{const.PTH_DIR}/UGCN/subset_DEMAND_1ch/random_node/{wave_type}2.pth")
    
    # train(clean_path=f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/train/clean",
    #       noisy_path=f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/train/{wave_type}",
    #       out_path=f"{const.PTH_DIR}/UGCN/subset_DEMAND_1ch/random_node/{wave_type}2.pth", batchsize=1)

    # test(mix_dir=f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/test/{wave_type}",
    #      out_dir=f"{const.OUTPUT_WAV_DIR}/UGCN/subset_DEMAND_1ch/random_node/subset_DEMAND_hoth_0505dB/{wave_type}",
    #      model_path=f"{const.PTH_DIR}/subset_DEMAND_hoth_0505dB/subset_DEMAND_hoth_0505dB.pth")
    
    test(mix_dir=f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/test/{wave_type}",
         out_dir=f"{const.OUTPUT_WAV_DIR}/UGCN/subset_DEMAND_1ch/random_node/test/{wave_type}",
         model_path=f"{const.PTH_DIR}/UGCN/subset_DEMAND_1ch/random_node/{wave_type}.pth")
    
    test(mix_dir=f"{const.MIX_DATA_DIR}/subset_DEMAND_hoth_1010dB_1ch/subset_DEMAND_hoth_1010dB_05sec_1ch/test/{wave_type}",
         out_dir=f"{const.OUTPUT_WAV_DIR}/UGCN/subset_DEMAND_1ch/random_node/test/{wave_type}2",
         model_path=f"{const.PTH_DIR}/UGCN/subset_DEMAND_1ch/random_node/{wave_type}2.pth")
