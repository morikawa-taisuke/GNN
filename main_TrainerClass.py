import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
import random

# TrainerClass.pyからEnhancementTrainerクラスをインポート
from TrainerClass import EnhancementTrainer
from models import (
    SpeqGCNNet, SpeqGATNet, SpeqGCNNet2, SpeqGATNet2,
    UGCN, UGAT, UGCN2, UGAT2
)
from DataLoader import AudioDataset, AudioDatasetTest

def main(model):
    # 設定ファイルの読み込み
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # TrainerClass.pyのEnhancementTrainerを使用して学習と推論を行う
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデル、損失関数、オプティマイザのインスタンス化
    model_name = config["model"]["name"]
    if model_name == "SpeqGCN": # モデル名をSpeqGCNに変更
        model = SpeqGCNNet(n_channels=config["model"]["num_mic"], n_classes=1, num_node=config["model"]["num_node"]).to(device)
    elif model_name == "SpeqGAT":
        model = SpeqGATNet(n_channels=config["model"]["num_mic"], n_classes=1, num_node=config["model"]["num_node"]).to(device)
    elif model_name == "SpeqGCN2":
        model = SpeqGCNNet2(n_channels=config["model"]["num_mic"], n_classes=1, num_node=config["model"]["num_node"]).to(device)
    elif model_name == "SpeqGAT2":
        model = SpeqGATNet2(n_channels=config["model"]["num_mic"], n_classes=1, num_node=config["model"]["num_node"]).to(device)
    elif model_name == "GCN":
        model = UGCN(n_channels=config["model"]["num_mic"], n_classes=1, num_node=config["model"]["num_node"]).to(device)
    elif model_name == "GAT":
        model = UGAT(n_channels=config["model"]["num_mic"], n_classes=1, num_node=config["model"]["num_node"]).to(device)
    elif model_name == "GCN2":
        model = UGCN2(n_channels=config["model"]["num_mic"], n_classes=1, num_node=config["model"]["num_node"]).to(device)
    elif model_name == "GAT2":
        model = UGAT2(n_channels=config["model"]["num_mic"], n_classes=1, num_node=config["model"]["num_node"]).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
        
    # config["model"]["domain"]に基づいてモデルを初期化
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["lr"])
    scheduler = None
    if config["optim"]["scheduler"] == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["optim"]["scheduler_params"]["mode"],
            factor=config["optim"]["scheduler_params"]["factor"],
            patience=config["optim"]["scheduler_params"]["patience"]
        )

    # データローダーのインスタンス化
    train_dataset = AudioDataset(noisy_dir=config["data"]["noisy_dir"], clean_dir=config["data"]["clean_dir"])# 学習用
    val_dataset = AudioDataset(noisy_dir=config["data"]["noisy_dir"], clean_dir=config["data"]["clean_dir"])# 検証用
    inference_dataset =  AudioDatasetTest# 推論用

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    # EnhancementTrainerのインスタンス化
    trainer = EnhancementTrainer(
        model=model,
        criterion=config["model"]["criterion"],  # 損失関数の指定
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        scheduler=scheduler
    )

    # 学習の実行
    trainer.train()

    print("\n--- Training finished. Starting inference ---")

    # 推論の実行
    output_wav_dir = Path(config["save_dir"]) / config["eval"]["output_dir"]
    output_csv_dir = Path(config["save_dir"]) / "eval_results_csv" # CSV出力ディレクトリを別途定義

    # 推論メソッドを呼び出す
    # calculate_metrics=Trueにすると評価指標も計算されます
    # output_wav_dirとoutput_csv_dirはPathオブジェクトではなく文字列で渡す必要があります
    # （TrainerClass.pyのinferenceメソッドの引数型ヒントがOptional[str]のため）
    results = trainer.inference(
        loader=inference_loader,
        output_wav_dir=str(output_wav_dir),
        output_csv_dir=str(output_csv_dir),
        calculate_metrics=True
    )

    if results:
        print("\n--- Inference and Evaluation Results ---")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

    print("\n--- Program finished ---")


if __name__ == "__main__":
    