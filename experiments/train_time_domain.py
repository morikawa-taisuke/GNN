import os
import time
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib import tenumerate

# モジュール読み込み時にパスが通るようおまじない（ルート実行想定）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.CsvDataset import CsvDataset, CsvInferenceDataset
from src.models.graph_utils import GraphConfig, NodeSelectionType, EdgeSelectionType
from src.mymodule import my_func, const, LossFunction, confirmation_GPU

# --- Wave Models (時間領域モデル群) ---
from src.models.ConvTasNet_models import enhance_ConvTasNet
from src.models.WaveUGNN import Wave_UGNN
from src.models.WaveUnet import Wave_UNet


def padding_tensor(tensor1, tensor2):
    """短い方のテンソルを末尾にゼロパディングして長さをそろえる"""
    len1 = tensor1.size(-1)
    len2 = tensor2.size(-1)
    max_len = max(len1, len2)

    pad1 = [0, max_len - len1]
    pad2 = [0, max_len - len2]

    return F.pad(tensor1, pad1), F.pad(tensor2, pad2)


def train(model: nn.Module,
          train_csv: str,
          val_csv: str,
          wave_type: str,
          out_path: str = "./RESULT/pth/result.pth",
          loss_type: str = "SISDR",
          batchsize: int = const.BATCHSIZE,
          checkpoint_path: str = None,
          train_count: int = const.EPOCH,
          earlystopping_threshold: int = 5,
          accumulation_steps: int = 4):
    """時間領域（Wave）モデル用の学習ループ"""
    device = confirmation_GPU.get_device()
    out_path = Path(out_path)
    out_name, out_dir = out_path.stem, out_path.parent
    
    writer = SummaryWriter(log_dir=f"{const.LOG_DIR}/{out_name}")
    now = my_func.get_now_time()
    csv_path = os.path.join(const.LOG_DIR, out_name, f"{out_name}_{now}.csv")
    my_func.make_dir(csv_path)
    
    with open(csv_path, "w", encoding="utf-8") as csv_file:
        csv_file.write(f"dataset,out_name,loss_func\n{train_csv},{out_path},{loss_type}\n")

    best_loss = np.inf
    earlystopping_count = 0

    train_dataset = CsvDataset(csv_path=train_csv, input_column_header=wave_type, max_length_sec=6)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

    val_dataset = CsvDataset(csv_path=val_csv, input_column_header=wave_type, max_length_sec=6)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=CsvDataset.collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = LossFunction.get_loss_computer(loss_type, device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"restart_training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 1

    print("=" * 20)
    print(f"Device: {device} | Domain: Time (Wave)")
    print(f"Out Path: {out_path} | Loss: {loss_type}")
    print("=" * 20)

    my_func.make_dir(out_dir)
    start_time = time.time()
    
    for epoch in range(start_epoch, train_count + 1):
        print(f"Train Epoch: {epoch}")
        model.train()
        model_loss_sum = 0
        optimizer.zero_grad()
        
        for i, (mix_data, target_data) in tenumerate(train_loader):
            mix_data, target_data = mix_data.to(device, dtype=torch.float32), target_data.to(device, dtype=torch.float32)

            # --- 時間領域モデルへの順伝播 ---
            estimate_data = model(mix_data)
            
            # --- データの整形と損失の計算 ---
            estimate_data, target_data = padding_tensor(estimate_data, target_data)

            model_loss = loss_func(estimate_data, target_data) / accumulation_steps
            model_loss.backward()
            model_loss_sum += model_loss.item() * accumulation_steps

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        # チェックポイント保存
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": model_loss_sum,
        }, f"{out_dir}/{out_name}_ckp.pth")
        
        writer.add_scalar(str(out_name[0]), model_loss_sum, epoch)
        print(f"[{epoch}] model_loss_sum: {model_loss_sum}")

        with open(csv_path, "a", encoding="utf-8") as out_file:
            out_file.write(f"{model_loss_sum}\n")

        # --- 検証 (Validation) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc="Validation")
            for mix_data, target_data in progress_bar_val:
                mix_data, target_data = mix_data.to(device, dtype=torch.float32), target_data.to(device, dtype=torch.float32)

                estimate_data = model(mix_data)
                estimate_data, target_data = padding_tensor(estimate_data, target_data)

                model_loss = loss_func(estimate_data, target_data)
                val_loss += model_loss.item()
                progress_bar_val.set_postfix({"loss": model_loss.item()})
                
            avg_val_loss = val_loss / len(val_loader)
            
        if avg_val_loss < best_loss:
            print(f"Validation loss improved ({best_loss:.6f} --> {avg_val_loss:.6f}). Saving model...")
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"{out_dir}/BEST_{out_name}.pth")
            earlystopping_count = 0
        else:
            earlystopping_count += 1
            print(f"Validation loss did not improve. Patience: {earlystopping_count}/{earlystopping_threshold}")

        if earlystopping_count >= earlystopping_threshold:
            print("Early stopping triggered. Training finished.")
            break

    # 最終出力モデルの保存
    torch.save(model.to(device).state_dict(), f"{out_dir}/{out_name}_{epoch}.pth")
    writer.close()
    print(f"Training completed. Total time: {(time.time() - start_time) / 3600.0:.2f}h")


def test(model: nn.Module, test_csv: str, wave_type: str, out_dir: str, model_path: str, prm: int = const.SR):
    """時間領域（Wave）モデル用の推論・テストループ"""
    device = confirmation_GPU.get_device()
    my_func.make_dir(out_dir)
    model_path = Path(model_path)
    model_dir, model_name = model_path.parent, model_path.stem

    model.load_state_dict(torch.load(os.path.join(model_dir, f"BEST_{model_name}.pth"), map_location=device))
    model.eval()

    dataset = CsvInferenceDataset(csv_path=test_csv, input_column_header=wave_type)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

    with torch.no_grad():
        for mix_data, mix_name in tqdm(dataset_loader):
            mix_data = mix_data.to(device, dtype=torch.float32)
            
            separate = model(mix_data)
            separate = separate.cpu().numpy().squeeze()
            
            # 正規化
            mix_max = torch.max(mix_data).cpu().numpy()
            data_to_write = separate / np.max(separate) * mix_max
            
            out_path = os.path.join(out_dir, f"{mix_name[0]}.wav")
            sf.write(out_path, data_to_write, prm)
            torch.cuda.empty_cache()


def main():
    """時間領域（Wave）モデル実行エントリーポイント"""
    device = confirmation_GPU.get_device()
    
    # 実行したいモデルをここで指定する
    model_architecture = "GCN" # 例: GCN, GAT, UNet, ConvTasNet
    
    num_mic = 1
    num_node = 32
    wave_types = ["noise_only", "reverb_only", "noise_reverb"]
    
    graph_config = GraphConfig(
        num_edges=num_node,
        node_selection=NodeSelectionType.ALL,
        edge_selection=EdgeSelectionType.RANDOM,
        bidirectional=True,
        temporal_window=4000
    )
    
    # モデルの動的ロード
    if model_architecture in ["GCN", "GAT"]:
        model = Wave_UGNN(gnn_type=model_architecture, graph_config=graph_config).to(device)
    elif model_architecture == "UNet":
        model = Wave_UNet(num_inputs=1, num_outputs=1).to(device)
    elif model_architecture == "ConvTasNet":
        model = enhance_ConvTasNet().to(device)
    else:
        raise ValueError(f"Unknown architecture: {model_architecture}")

    dir_name = "Random_Dataset_VCTK_DEMAND_1ch"
    loss_type = "SISDR"
    prefix = f"Wave_{model_architecture}"
    
    for wave_type in wave_types:
        out_name = f"{prefix}_{wave_type}_{num_node}node"
        
        train(
            model=model,
            train_csv=f"{const.MIX_DATA_DIR}/{dir_name}/train.csv",
            val_csv=f"{const.MIX_DATA_DIR}/{dir_name}/val.csv",
            wave_type=wave_type,
            out_path=f"{const.PTH_DIR}/{dir_name}/{prefix}/{out_name}.pth",
            loss_type=loss_type,
            batchsize=2,
            accumulation_steps=8
        )
        
        test(
            model=model,
            test_csv=f"{const.MIX_DATA_DIR}/{dir_name}/test.csv",
            wave_type=wave_type,
            out_dir=f"{const.OUTPUT_WAV_DIR}/{dir_name}/{prefix}/{out_name}",
            model_path=f"{const.PTH_DIR}/{dir_name}/{prefix}/{out_name}.pth"
        )

if __name__ == "__main__":
    main()
