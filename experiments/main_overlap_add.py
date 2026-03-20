import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR
# Import torchmetrics for loss functions
from torchmetrics.regression import MeanSquaredError as MSE
from tqdm import tqdm
from tqdm.contrib import tenumerate

from All_evaluation import main as evaluation
# from All_evaluation_torch import main as evaluation
from UGNNNet_DatasetClass import AudioDataset, AudioDataset_test
from models.ConvTasNet_models import enhance_ConvTasNet
from models.GNN import UGCN, UGAT, UGCN2, UGAT2
from models.GNN_encoder import GNNEncoder
from models.wave_unet import U_Net
from mymodule import my_func, const
from mymodule.confirmation_GPU import get_device

# CUDA縺ｮ繝｡繝｢繝ｪ邂｡逅・ｨｭ螳・
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CUDA縺ｮ蜿ｯ逕ｨ諤ｧ繧偵メ繧ｧ繝・け
device = get_device()  # 菴ｿ逕ｨ蜿ｯ閭ｽ縺ｪ繝・ヰ繧､繧ｹ繧貞叙蠕・
# device = "mps"
print(f"Using device: {device}")


def padding_tensor(tensor1, tensor2):
    """
    譛蠕後・谺｡蜈・ｼ井ｾ・ 譎らｳｻ蛻鈴聞・峨′逡ｰ縺ｪ繧・縺､縺ｮ繝・Φ繧ｽ繝ｫ縺ｫ蟇ｾ縺励※縲・
    遏ｭ縺・婿繧呈忰蟆ｾ縺ｫ繧ｼ繝ｭ繝代ョ繧｣繝ｳ繧ｰ縺励※髟ｷ縺輔ｒ縺昴ｍ縺医ｋ縲・

    Args:
        tensor1, tensor2 (torch.Tensor): 莉ｻ諢上・谺｡蜈・焚縺ｮ繝・Φ繧ｽ繝ｫ

    Returns:
        padded_tensor1, padded_tensor2 (torch.Tensor)
    """
    len1 = tensor1.size(-1)
    len2 = tensor2.size(-1)
    max_len = max(len1, len2)

    pad1 = [0, max_len - len1]  # 譛蠕後・谺｡蜈・□縺代ヱ繝・ぅ繝ｳ繧ｰ
    pad2 = [0, max_len - len2]

    padded_tensor1 = F.pad(tensor1, pad1)
    padded_tensor2 = F.pad(tensor2, pad2)

    return padded_tensor1, padded_tensor2


def train(
    model: nn.Module,
    mix_dir: str,
    clean_dir: str,
    out_path: str = "./RESULT/pth/result.pth",
    loss_func: str = "stft_MSE",
    batchsize: int = const.BATCHSIZE,
    checkpoint_path: str = None,
    train_count: int = const.EPOCH,
    earlystopping_threshold: int = 5,
):
    """GPU縺ｮ險ｭ螳・""
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU縺御ｽｿ縺医ｌ縺ｰ菴ｿ縺・
    """ 縺昴・莉悶・險ｭ螳・"""
    out_path = Path(out_path)  # path蝙九↓螟画鋤
    out_name, out_dir = out_path.stem, out_path.parent  # 繝輔ぃ繧､繝ｫ蜷阪→繝・ぅ繝ｬ繧ｯ繝医Μ繧貞・髮｢
    writer = SummaryWriter(
        log_dir=f"{const.LOG_DIR}\\{out_name}"
    )  # log縺ｮ菫晏ｭ伜・縺ｮ謖・ｮ・"tensorboard --logdir ./logs"縺ｧ遒ｺ隱阪〒縺阪ｋ)
    now = my_func.get_now_time()
    csv_path = os.path.join(
        const.LOG_DIR, out_name, f"{out_name}_{now}.csv"
    )  # CSV繝輔ぃ繧､繝ｫ縺ｮ繝代せ
    my_func.make_dir(csv_path)
    with open(csv_path, "w") as csv_file:  # 繝輔ぃ繧､繝ｫ繧ｪ繝ｼ繝励Φ
        csv_file.write(f"dataset,out_name,loss_func\n{mix_dir},{out_path},{loss_func}")

    """ Early_Stopping縺ｮ險ｭ螳・"""
    best_loss = np.inf  # 謳榊､ｱ髢｢謨ｰ縺ｮ譛蟆丞喧縺檎岼逧・・蝣ｴ蜷茨ｼ悟・繧√・best_loss繧堤┌髯仙､ｧ縺ｫ縺吶ｋ
    earlystopping_count = 0

    """ Load dataset 繝・・繧ｿ繧ｻ繝・ヨ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ """
    dataset = AudioDataset(
        clean_audio_dir=clean_dir, noisy_audio_dir=mix_dir
    )  # 繝・・繧ｿ繧ｻ繝・ヨ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
    dataset_loader = DataLoader(
        dataset, batch_size=batchsize, shuffle=True, pin_memory=True
    )

    # print(f"\nmodel:{model}\n")                           # 繝｢繝・Ν縺ｮ繧｢繝ｼ繧ｭ繝・け繝√Ε縺ｮ蜃ｺ蜉・
    """ 譛驕ｩ蛹夜未謨ｰ縺ｮ險ｭ螳・"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer繧帝∈謚・Adam)
    if loss_func == "SISDR":
        loss_metric = SISDR().to(device)
    elif loss_func == "wave_MSE" or loss_func == "stft_MSE":
        loss_metric = MSE().to(device)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

    """ 繝√ぉ繝・け繝昴う繝ｳ繝医・險ｭ螳・"""
    if checkpoint_path != None:
        print("restart_training")
        checkpoint = torch.load(checkpoint_path)  # checkpoint縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )  # 蟄ｦ鄙帝比ｸｭ縺ｮ繝｢繝・Ν縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )  # 繧ｪ繝励ユ繧｣繝槭う繧ｶ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
        # optimizer縺ｮstate繧堤樟蝨ｨ縺ｮdevice縺ｫ遘ｻ縺吶ゅ％繧後ｒ縺励↑縺・→縲∽ｿ晏ｭ伜燕蠕後〒device縺ｮ荳肴紛蜷医′襍ｷ縺薙ｋ蜿ｯ閭ｽ諤ｧ縺後≠繧九・
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
    else:
        start_epoch = 1

    """ 蟄ｦ鄙偵・險ｭ螳壹ｒ蜃ｺ蜉・"""
    print("====================")
    print("device: ", device)
    print("out_path: ", out_path)
    print("dataset: ", mix_dir)
    print("loss_func: ", loss_func)
    print("====================")

    my_func.make_dir(out_dir)
    model.train()  # 蟄ｦ鄙偵Δ繝ｼ繝峨↓險ｭ螳・

    start_time = time.time()  # 譎る俣繧呈ｸｬ螳・
    epoch = 0
    for epoch in range(start_epoch, train_count + 1):  # 蟄ｦ鄙貞屓謨ｰ
        print("Train Epoch:", epoch)  # 蟄ｦ鄙貞屓謨ｰ縺ｮ陦ｨ遉ｺ
        model_loss_sum = 0  # 邱乗錐螟ｱ縺ｮ蛻晄悄蛹・
        for _, (mix_data, target_data) in tenumerate(dataset_loader):
            # 繝・・繧ｿ繧竪PU縺ｫ遘ｻ蜍・
            mix_data, target_data = mix_data.to(device), target_data.to(device)

            """ 蜍ｾ驟阪・繝ｪ繧ｻ繝・ヨ """
            optimizer.zero_grad()  # optimizer縺ｮ蛻晄悄蛹・

            """ 繝・・繧ｿ縺ｮ謨ｴ蠖｢ """
            # 繧ｿ繧､繝励ｒ螟画鋤 int16竊断loat32
            mix_data = mix_data.to(torch.float32)
            target_data = target_data.to(torch.float32)

            """ 竊凪・竊・繧ｪ繝ｼ繝舌・繝ｩ繝・・繧｢繝峨・蟆主・ 竊凪・竊・"""
            batchsize, num_channels, signal_length = (
                mix_data.shape
            )  # 繝・・繧ｿ縺ｮ蠖｢迥ｶ繧貞叙蠕・
            flame_size = int(const.SR * 0.1)  # 繝輔Ξ繝ｼ繝繧ｵ繧､繧ｺ・・00ms・・
            hop_size = flame_size // 2  # 繝帙ャ繝励し繧､繧ｺ縺ｯ繝輔ぅ繝ｫ繧ｿ髟ｷ縺ｮ蜊雁・
            # 繧ｼ繝ｭ繝代ョ繧｣繝ｳ繧ｰ
            mix_padded = torch.cat(
                (
                    torch.zeros(batchsize, num_channels, hop_size, device=device),
                    mix_data,
                ),
                dim=2,
            ).requires_grad_(True)
            target_padded = torch.cat(
                (
                    torch.zeros(batchsize, num_channels, hop_size, device=device),
                    target_data,
                ),
                dim=2,
            ).requires_grad_(True)
            estimation = torch.zeros(mix_padded.shape, device=device)  # 蜃ｺ蜉帷畑縺ｮ驟榊・
            num_flame = mix_padded.shape[-1] // hop_size  # 繝輔Ξ繝ｼ繝謨ｰ縺ｮ險育ｮ・

            for i in range(num_flame):
                start = i * hop_size
                end = start + flame_size

                # 繝悶Ο繝・け繧貞叙蠕・
                flame = mix_padded[:, :, start:end]

                # 遯薙°縺・
                window = torch.hann_window(
                    flame_size, requires_grad=True, device=device
                )
                if flame.shape[-1] != flame_size:
                    # 繝輔Ξ繝ｼ繝繧ｵ繧､繧ｺ縺檎焚縺ｪ繧句ｴ蜷医・縲√ヵ繝ｬ繝ｼ繝繧ｵ繧､繧ｺ縺ｫ蜷医ｏ縺帙※蛻・ｊ隧ｰ繧√ｋ
                    flame = F.pad(
                        flame,
                        (0, flame_size - flame.shape[-1]),
                        mode="constant",
                        value=0,
                    )
                flame_windowed = flame * window  # 繝上ル繝ｳ繧ｰ遯薙ｒ驕ｩ逕ｨ
                """ 繝｢繝・Ν縺ｫ騾壹☆(莠域ｸｬ蛟､縺ｮ險育ｮ・ """
                # print("model_input", mix_data.shape)
                estimate_flame = model(flame_windowed)  # 繝｢繝・Ν縺ｫ騾壹☆

                # 蜃ｺ蜉帙↓邨先棡繧貞刈邂・
                end_index = min(
                    estimation.shape[-1], end
                )  # 驟榊・縺ｮ髟ｷ縺輔ｒ雜・∴縺ｪ縺・ｈ縺・↓隱ｿ謨ｴ
                estimation[:, :, start:end_index] = (
                    estimation[:, :, start:end_index]
                    + estimate_flame[:, :, : end_index - start]
                )
            """ 竊鯛・竊・繧ｪ繝ｼ繝舌・繝ｩ繝・・繧｢繝峨・蟆主・ 竊鯛・竊・"""

            """ 繝・・繧ｿ縺ｮ謨ｴ蠖｢ """
            # print("estimation:", estimate_data.shape)
            # print("target:", target_data.shape)
            estimation, target_padded = padding_tensor(estimation, target_padded)

            """ 謳榊､ｱ縺ｮ險育ｮ・"""
            model_loss = 0
            match loss_func:
                case "SISDR":
                    model_loss = -1 * loss_metric(estimation, target_data)
                case "wave_MSE":
                    model_loss = loss_metric(
                        estimation, target_padded
                    )  # 譎る俣豕｢蠖｢荳翫〒MSE縺ｫ繧医ｋ謳榊､ｱ髢｢謨ｰ縺ｮ險育ｮ・
                case "stft_MSE":
                    """蜻ｨ豕｢謨ｰ霆ｸ縺ｫ螟画鋤"""
                    stft_estimate_data = torch.stft(
                        estimation[0], n_fft=1024, return_complex=False
                    )
                    stft_target_data = torch.stft(
                        target_padded[0], n_fft=1024, return_complex=False
                    )
                    model_loss = loss_metric(
                        stft_estimate_data, stft_target_data
                    )  # 譎る俣蜻ｨ豕｢謨ｰ荳凱SE縺ｫ繧医ｋ謳榊､ｱ縺ｮ險育ｮ・

            # print(f"model_loss: {model_loss.item()}")  # 謳榊､ｱ縺ｮ蜃ｺ蜉・
            model_loss_sum += model_loss  # 謳榊､ｱ縺ｮ蜉邂・

            """ 蠕悟・逅・"""
            model_loss.backward()  # 隱､蟾ｮ騾・ｼ晄成
            optimizer.step()  # 蜍ｾ驟阪・譖ｴ譁ｰ

            del (
                mix_data,
                target_data,
                estimation,
                model_loss,
            )  # 菴ｿ逕ｨ縺励※縺・↑縺・､画焚縺ｮ蜑企勁 estimate_data,
            torch.cuda.empty_cache()  # 繝｡繝｢繝ｪ縺ｮ隗｣謾ｾ 1iteration縺斐→縺ｫ隗｣謾ｾ

        """ 繝√ぉ繝・け繝昴う繝ｳ繝医・菴懈・ """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": model_loss_sum,
            },
            f"{out_dir}/{out_name}_ckp.pth",
        )

        writer.add_scalar(str(out_name[0]), model_loss_sum, epoch)
        print(f"[{epoch}]model_loss_sum:{model_loss_sum}")  # 謳榊､ｱ縺ｮ蜃ｺ蜉・

        torch.cuda.empty_cache()  # 繝｡繝｢繝ｪ縺ｮ隗｣謾ｾ 1iteration縺斐→縺ｫ隗｣謾ｾ
        with open(csv_path, "a") as out_file:  # 繝輔ぃ繧､繝ｫ繧ｪ繝ｼ繝励Φ
            out_file.write(f"{model_loss_sum}\n")  # 譖ｸ縺崎ｾｼ縺ｿ

        """ Early_Stopping 縺ｮ蛻､譁ｭ """
        # best_loss縺ｨmodel_loss_sum繧呈ｯ碑ｼ・
        if model_loss_sum < best_loss:  # model_loss縺ｮ縺ｻ縺・′蟆上＆縺・ｴ蜷・
            print(f"{epoch:3} [epoch] | {model_loss_sum:.6} <- {best_loss:.6}")
            torch.save(
                model.to(device).state_dict(), f"{out_dir}/BEST_{out_name}.pth"
            )  # 蜃ｺ蜉帙ヵ繧｡繧､繝ｫ縺ｮ菫晏ｭ・
            best_loss = model_loss_sum  # best_loss縺ｮ螟画峩
            earlystopping_count = 0
            estimation = estimation.cpu()
            estimation = estimation.detach().numpy()
            estimation = estimation.squeeze()  # (1, 1, length) -> (length,)
            sf.write("./RESULT/BEST.wav", estimation, const.SR)

        else:
            earlystopping_count += 1
            if (epoch > 100) and (earlystopping_count > earlystopping_threshold):
                break
        if epoch == 100:
            torch.save(
                model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth"
            )  # 蜃ｺ蜉帙ヵ繧｡繧､繝ｫ縺ｮ菫晏ｭ・

    """ 蟄ｦ鄙偵Δ繝・Ν(pth繝輔ぃ繧､繝ｫ)縺ｮ蜃ｺ蜉・"""
    print("model save")
    torch.save(
        model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth"
    )  # 蜃ｺ蜉帙ヵ繧｡繧､繝ｫ縺ｮ菫晏ｭ・

    writer.close()

    """ 蟄ｦ鄙呈凾髢薙・險育ｮ・"""
    time_end = time.time()  # 迴ｾ蝨ｨ譎る俣縺ｮ蜿門ｾ・
    time_sec = time_end - start_time  # 邨碁℃譎る俣縺ｮ險育ｮ・sec)
    time_h = float(time_sec) / 3600.0  # sec->hour
    print(f"time・嘴str(time_h)}h")  # 蜃ｺ蜉・


def test(
    model: nn.Module, mix_dir: str, out_dir: str, model_path: str, prm: int = const.SR
):
    # filelist_mixdown = my_func.get_file_list(mix_dir)
    # print('number of mixdown file', len(filelist_mixdown))

    # 繝・ぅ繝ｬ繧ｯ繝医Μ繧剃ｽ懈・
    my_func.make_dir(out_dir)
    model_path = Path(model_path)  # path蝙九↓螟画鋤
    model_dir, model_name = (
        model_path.parent,
        model_path.stem,
    )  # 繝輔ぃ繧､繝ｫ蜷阪→繝・ぅ繝ｬ繧ｯ繝医Μ繧貞・髮｢

    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, f"BEST_{model_name}.pth"), map_location=device
        )
    )
    model.eval()

    dataset = AudioDataset_test(mix_dir)  # 繝・・繧ｿ繧ｻ繝・ヨ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

    for mix_data, mix_name in tqdm(dataset_loader):
        mix_data = mix_data.to(device)  # 繝・・繧ｿ繧竪PU縺ｫ遘ｻ蜍・
        mix_data = mix_data.to(torch.float32)  # 繝・・繧ｿ縺ｮ蝙九ｒ螟画鋤 int16竊断loat32

        mix_max = torch.max(mix_data)  # 譛螟ｧ蛟､縺ｮ蜿門ｾ・

        """ 竊凪・竊・繧ｪ繝ｼ繝舌・繝ｩ繝・・繧｢繝峨・蟆主・ 竊凪・竊・"""
        batchsize, num_channels, _ = mix_data.shape  # 繝・・繧ｿ縺ｮ蠖｢迥ｶ繧貞叙蠕・
        flame_size = int(const.SR * 0.1)  # 繝輔Ξ繝ｼ繝繧ｵ繧､繧ｺ・・00ms・・
        hop_size = flame_size // 2  # 繝帙ャ繝励し繧､繧ｺ縺ｯ繝輔ぅ繝ｫ繧ｿ髟ｷ縺ｮ蜊雁・
        mix_padded = torch.cat(
            (
                torch.zeros(batchsize, num_channels, hop_size, device=device),
                mix_data,
            ),
            dim=2,
        ).requires_grad_(
            True
        )  # 蜈･蜉帑ｿ｡蜿ｷ縺ｮ蜑阪↓0繧定ｿｽ蜉
        estimation = torch.zeros(mix_padded.shape, device=device)  # 蜃ｺ蜉帷畑縺ｮ驟榊・
        num_flame = mix_padded.shape[-1] // hop_size  # 繝輔Ξ繝ｼ繝謨ｰ縺ｮ險育ｮ・

        for i in range(num_flame):
            start = i * hop_size
            end = start + flame_size

            # 繝悶Ο繝・け繧貞叙蠕・
            flame = mix_padded[:, :, start:end]

            # 遯薙°縺・
            window = torch.hann_window(flame_size, requires_grad=True, device=device)
            if flame.shape[-1] != flame_size:
                # 繝輔Ξ繝ｼ繝繧ｵ繧､繧ｺ縺檎焚縺ｪ繧句ｴ蜷医・縲√ヵ繝ｬ繝ｼ繝繧ｵ繧､繧ｺ縺ｫ蜷医ｏ縺帙※蛻・ｊ隧ｰ繧√ｋ
                flame = F.pad(
                    flame,
                    (0, flame_size - flame.shape[-1]),
                    mode="constant",
                    value=0,
                )
            flame_windowed = (flame * window).to(device)  # 繝上ル繝ｳ繧ｰ遯薙ｒ驕ｩ逕ｨ
            """ 繝｢繝・Ν縺ｫ騾壹☆(莠域ｸｬ蛟､縺ｮ險育ｮ・ """
            # print("model_input", mix_data.shape)
            estimate_flame = model(flame_windowed)  # 繝｢繝・Ν縺ｫ騾壹☆

            # 蜃ｺ蜉帙↓邨先棡繧貞刈邂・
            end_index = min(estimation.shape[-1], end)  # 驟榊・縺ｮ髟ｷ縺輔ｒ雜・∴縺ｪ縺・ｈ縺・↓隱ｿ謨ｴ
            estimation[:, :, start:end_index] = (
                estimation[:, :, start:end_index]
                + estimate_flame[:, :, : end_index - start]
            )
        """ 竊鯛・竊・繧ｪ繝ｼ繝舌・繝ｩ繝・・繧｢繝峨・蟆主・ 竊鯛・竊・"""

        estimation = estimation.cpu()
        estimation = estimation.detach().numpy()

        # 繝｢繝・Ν縺ｮ蜃ｺ蜉帙′ (1, 1, length) 縺ｨ莉ｮ螳・
        data_to_write = estimation.squeeze()

        # 豁｣隕丞喧
        mix_max = torch.max(mix_data)  # mix_wave縺ｮ譛螟ｧ蛟､繧貞叙蠕・
        data_to_write = (
            data_to_write / np.max(data_to_write) * mix_max.cpu().detach().numpy()
        )

        # 菫晏ｭ・
        # 繝輔ぃ繧､繝ｫ蜷阪→繝輔か繝ｫ繝蜷阪ｒ邨仙粋縺励※繝代せ譁・ｭ怜・繧剃ｽ懈・
        out_path = os.path.join(out_dir, (mix_name[0] + ".wav"))
        # 豺ｷ蜷医ョ繝ｼ繧ｿ繧剃ｿ晏ｭ・
        sf.write(out_path, data_to_write, prm)
        torch.cuda.empty_cache()  # 繝｡繝｢繝ｪ縺ｮ隗｣謾ｾ 1髻ｳ螢ｰ縺斐→縺ｫ隗｣謾ｾ


if __name__ == "__main__":
    """繝｢繝・Ν縺ｮ險ｭ螳・""
    num_mic = 1  # 繝槭う繧ｯ縺ｮ謨ｰ
    num_node = 16  # 繝弱・繝峨・謨ｰ
    model_list = [
        "GCNEncoder",
        "GATEncoder",
    ]  # 繝｢繝・Ν縺ｮ遞ｮ鬘・ "UGCN", "UGCN2", "UGAT", "UGAT2", "ConvTasNet", "UNet", "GCNEncoder", "GATEncoder"
    wave_types = [
        "noise_reverb",
        "reverbe_only",
        "noise_only",
    ]  # 蜈･蜉帑ｿ｡蜿ｷ縺ｮ遞ｮ鬘・(noise_only, reverbe_only, noise_reverb)

    for model_type in model_list:
        if model_type == "UGCN":
            model = UGCN(n_channels=num_mic, num_node=num_node).to(device)
        elif model_type == "UGAT":
            model = UGAT(
                n_channels=num_mic,
                num_node=num_node,
                gat_heads=4,
                gat_dropout=0.6,
            ).to(device)
        elif model_type == "UGCN2":
            model = UGCN2(n_channels=num_mic, num_node=num_node).to(device)
        elif model_type == "UGAT2":
            model = UGAT2(
                n_channels=num_mic,
                num_node=num_node,
                gat_heads=4,
                gat_dropout=0.6,
            ).to(device)
        elif model_type == "ConvTasNet":
            model = enhance_ConvTasNet().to(device)
        elif model_type == "UNet":
            model = U_Net().to(device)
        elif model_type == "GCNEncoder":
            model = GNNEncoder(
                n_channels=num_mic, gnn_type="GCN", num_node=num_node
            ).to(device)
        elif model_type == "GATEncoder":
            model = GNNEncoder(
                n_channels=num_mic, gnn_type="GAT", num_node=num_node
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        for wave_type in wave_types:
            out_name = f"{model_type}_{wave_type}_{num_node}node"  # 蜃ｺ蜉帙ヵ繧｡繧､繝ｫ蜷・
            # train(
            #     model=model,
            #     mix_dir=f"{const.MIX_DATA_DIR}/GNN/subset_DEMAND_hoth_5dB_500msec/train/{wave_type}",
            #     clean_dir=f"{const.MIX_DATA_DIR}/GNN/subset_DEMAND_hoth_5dB_500msec/train/clean",
            #     out_path=f"{const.PTH_DIR}/{model_type}/subset_DEMAND_hoth_5dB_500msec/{out_name}.pth",
            #     batchsize=1,
            #     loss_func="SISDR",
            #     checkpoint_path=None,
            #     train_count=const.EPOCH,
            #     earlystopping_threshold=5,
            # )

            test(
                model=model,
                mix_dir=f"{const.MIX_DATA_DIR}/GNN/subset_DEMAND_hoth_5dB_500msec/test/{wave_type}",
                out_dir=f"{const.OUTPUT_WAV_DIR}/{model_type}/subset_DEMAND_hoth_5dB_500msec/{out_name}_overlap",
                model_path=f"{const.PTH_DIR}/{model_type}/subset_DEMAND_hoth_5dB_500msec/{out_name}.pth",
            )

            evaluation(
                target_dir=f"{const.MIX_DATA_DIR}/GNN/subset_DEMAND_hoth_5dB_500msec/test/clean",
                estimation_dir=f"{const.OUTPUT_WAV_DIR}/{model_type}/subset_DEMAND_hoth_5dB_500msec/{out_name}_overlap",
                out_path=f"{const.EVALUATION_DIR}/{out_name}_overlap.csv",
                # device=torch.device("cpu"),
            )
