# TrainerClass.py

import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple

from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio as SI_SDR,
    PerceptualEvaluationSpeechQuality as PESQ,
    ShortTimeObjectiveIntelligibility as STOI,
)

# TensorBoard用のライター
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class EnhancementTrainer:
    """
    音声強調モデルの学習、推論、評価を管理するトレーナークラス。

    Attributes:
        model (torch.nn.Module): 学習・評価対象のPyTorchモデル。
        criterion (torch.nn.Module): 損失関数。
        optimizer (torch.optim.Optimizer): オプティマイザ。
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学習率スケジューラ。
        train_loader (torch.utils.data.DataLoader): 学習用データローダー。
        val_loader (torch.utils.data.DataLoader): 検証用データローダー。
        config (Dict[str, Any]): 学習と評価に関する設定。
        device (torch.device): 計算に使用するデバイス。
        domain (str): モデルが扱うデータドメイン ('time' or 'frequency')。
        save_dir (Path): チェックポイントやログの保存先ディレクトリ。
        writer (SummaryWriter, optional): TensorBoard用のロガー。
        best_val_loss (float): これまでの最小の検証損失。
        start_epoch (int): 学習を開始するエポック番号。
    """

    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            config: Dict[str, Any],
            device: torch.device,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        EnhancementTrainerのインスタンスを初期化する。
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # 設定項目を読み込む
        self.domain = self.config.get("domain", "time")
        if self.domain not in ["time", "frequency"]:
            raise ValueError(f"Unsupported domain: {self.domain}. Must be 'time' or 'frequency'.")

        self.epochs = self.config.get("epochs", 100)
        self.save_dir = Path(self.config.get("save_dir", "experiments"))
        self.sample_rate = self.config.get("sample_rate", 16000)

        # STFTパラメータ (周波数ドメインモデルの場合)
        if self.domain == "frequency":
            stft_params = self.config.get("stft_params")
            if not stft_params:
                raise ValueError("stft_params must be provided for frequency domain models.")
            self.n_fft = stft_params["n_fft"]
            self.hop_length = stft_params["hop_length"]
            self.win_length = stft_params["win_length"]

        # 保存ディレクトリとロガーの準備
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if SummaryWriter:
            self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))
        else:
            self.writer = None
            print("Warning: tensorboard not found. Logging will be disabled.")

        # 状態変数の初期化
        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.global_step = 0

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """時間領域の波形を複素スペクトログラムに変換する。"""
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(self.device),
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """複素スペクトログラムを時間領域の波形に逆変換する。"""
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(self.device),
            length=length,
        )

    def _prepare_input(self, noisy_waveform: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """モデルのドメインに応じて入力を準備する。"""
        if self.domain == 'frequency':
            noisy_spec = self._stft(noisy_waveform)
            # 位相情報を後で使うために保持
            phase = noisy_spec.angle()
            # モデル入力は通常、振幅スペクトログラム
            model_input = noisy_spec.abs()
            return model_input, phase
        else:  # time domain
            return noisy_waveform, None

    def _process_output(self, model_output: torch.Tensor, original_len: int, phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """モデルの出力から最終的な音声波形を生成する。"""
        if self.domain == 'frequency':
            if phase is None:
                raise ValueError("Phase information is required for ISTFT in frequency domain models.")
            # モデル出力（振幅）と元の位相を組み合わせて複素スペクトログラムを再構築
            complex_spec = model_output * torch.exp(1j * phase)
            enhanced_waveform = self._istft(complex_spec, length=original_len)
        else:  # time domain
            enhanced_waveform = model_output

        return enhanced_waveform

    def _train_epoch(self, epoch: int):
        """1エポック分の学習処理。"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Training]")
        for noisy, clean in progress_bar:
            noisy, clean = noisy.to(self.device), clean.to(self.device)

            self.optimizer.zero_grad()

            # モデルへの入力準備
            model_input, _ = self._prepare_input(noisy)

            # モデルのターゲット準備
            if self.domain == 'frequency':
                target = self._stft(clean).abs()
            else:
                target = clean

            # フォワードパス
            output = self.model(model_input)

            # 損失計算
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            progress_bar.set_postfix(loss=loss.item())
            if self.writer:
                self._log_metrics({'train/loss_step': loss.item()}, self.global_step, 'train')

        avg_loss = total_loss / len(self.train_loader)
        if self.writer:
            self._log_metrics({'train/loss_epoch': avg_loss}, epoch, 'train')
        print(f"Epoch {epoch}/{self.epochs} [Training] Avg Loss: {avg_loss:.4f}")

    def _validate_epoch(self, epoch: int) -> float:
        """1エポック分の検証処理。"""
        self.model.eval()
        total_loss = 0.0

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Validation]")
        with torch.no_grad():
            for noisy, clean in progress_bar:
                noisy, clean = noisy.to(self.device), clean.to(self.device)

                # モデルへの入力準備
                model_input, _ = self._prepare_input(noisy)

                # モデルのターゲット準備
                if self.domain == 'frequency':
                    target = self._stft(clean).abs()
                else:
                    target = clean

                # フォワードパス
                output = self.model(model_input)

                # 損失計算
                loss = self.criterion(output, target)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)
        if self.writer:
            self._log_metrics({'val/loss_epoch': avg_loss}, epoch, 'validation')
        print(f"Epoch {epoch}/{self.epochs} [Validation] Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        """学習ループ全体を管理する。"""
        print("Starting training...")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)    # 学習エポックの実行
            val_loss = self._validate_epoch(epoch)  # 検証エポックの実行

            if self.scheduler:
                # ReduceLROnPlateauの場合は損失を渡す
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best model found at epoch {epoch} with validation loss: {val_loss:.4f}")

            self._save_checkpoint(epoch, is_best)

        if self.writer:
            self.writer.close()
        print("Training finished.")

    def inference(
            self,
            loader: torch.utils.data.DataLoader,
            output_wav_dir: Optional[str] = None,
            output_csv_dir: Optional[str] = None,
            calculate_metrics: bool = True,
    ) -> Optional[Dict[str, float]]:
        """
        推論と客観評価をまとめて実行する。

        このメソッドは、データセットに対してモデルの推論を行い、オプションで
        強調された音声をファイルに保存したり、客観評価指標を計算・保存したりできる。

        Args:
            loader (torch.utils.data.DataLoader): 推論・評価用のデータローダー。
                返すタプルの内容は、実行するタスクに依存します。
                - 評価 (`calculate_metrics=True`) と WAV/CSV保存の両方を行う場合:
                  `(noisy_tensor, clean_tensor, filename_list)` を返す必要があります。
                - 評価のみを行う場合:
                  `(noisy_tensor, clean_tensor)` を返す必要があります。
                - WAV保存のみを行う場合:
                  `(noisy_tensor, filename_list)` を返す必要があります。
            output_wav_dir (Optional[str], optional): 強調された音声の保存先ディレクトリ。
                Noneの場合、音声ファイルは保存されない。 Defaults to None.
            output_csv_dir (Optional[str], optional): ファイルごとの客観評価を記録したCSVファイルの保存先ディレクトリ。
                `calculate_metrics=True` の場合のみ有効。 Defaults to None.
            calculate_metrics (bool, optional): 客観評価指標 (SI-SDR, STOI, PESQ) を
                計算するかどうか。 Defaults to True.

        Returns:
            Optional[Dict[str, float]]: `calculate_metrics=True` の場合、データセット全体の
                                        平均評価結果の辞書を返す。それ以外の場合は None を返す。
        """
        if not output_wav_dir and not calculate_metrics:
            print("Warning: Both output_wav_dir and calculate_metrics are disabled. No action will be performed.")
            return None

        if output_csv_dir and not calculate_metrics:
            print("Warning: `output_csv_dir` is specified, but `calculate_metrics` is False. No CSV will be generated.")
            output_csv_dir = None

        self.model.eval()

        # --- 初期設定 ---
        if output_wav_dir:
            output_path = Path(output_wav_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Starting inference, saving files to {output_path}...")

        per_file_results = []
        if calculate_metrics:
            if self.sample_rate not in [8000, 16000]:
                print(
                    f"Warning: PESQ requires 8kHz or 16kHz sample rate. Current is {self.sample_rate}. PESQ will be skipped.")
                use_pesq = False
            else:
                use_pesq = True
                mode = 'wb' if self.sample_rate == 16000 else 'nb'

            if output_csv_dir:
                output_csv_path = Path(output_csv_dir)
                output_csv_path.mkdir(parents=True, exist_ok=True)
                print(f"Evaluation results will be saved to {output_csv_path}")

        # --- メインループ ---
        progress_bar_desc = "Inference"
        if calculate_metrics: progress_bar_desc = "Evaluation"
        if output_wav_dir: progress_bar_desc = f"Inference to {Path(output_wav_dir).name}"
        if calculate_metrics and output_wav_dir: progress_bar_desc = f"Eval & Save to {Path(output_wav_dir).name}"

        progress_bar = tqdm(loader, desc=progress_bar_desc)

        with torch.no_grad():
            for batch in progress_bar:
                # --- データのアンパック ---
                if calculate_metrics and (output_wav_dir or output_csv_dir):
                    noisy, clean, filenames = batch
                elif calculate_metrics:
                    noisy, clean = batch
                    filenames = None
                elif output_wav_dir:
                    noisy, filenames = batch
                    clean = None
                else:
                    continue

                noisy = noisy.to(self.device)
                original_len = noisy.shape[-1]

                # --- 推論実行 ---
                model_input, phase = self._prepare_input(noisy)
                model_output = self.model(model_input)
                enhanced = self._process_output(model_output, original_len, phase)

                # --- 評価指標の計算 (ファイルごと) ---
                if calculate_metrics:
                    clean = clean.to(self.device)
                    # バッチ内の各ファイルに対して計算
                    for i in range(enhanced.shape[0]):
                        enh_i, cln_i = enhanced[i:i + 1], clean[i:i + 1]

                        si_sdr_val = scale_invariant_signal_distortion_ratio(enh_i, cln_i).item()
                        stoi_val = short_time_objective_intelligibility(enh_i, cln_i, self.sample_rate,
                                                                        extended=False).item()

                        pesq_val = None
                        if use_pesq:
                            try:
                                pesq_val = perceptual_evaluation_speech_quality(enh_i, cln_i, self.sample_rate,
                                                                                mode).item()
                            except Exception as e:
                                print(f"Could not compute PESQ for a file, skipping. Error: {e}")

                        file_metric = {"SI-SDR": si_sdr_val, "STOI": stoi_val}
                        if pesq_val is not None:
                            file_metric["PESQ"] = pesq_val

                        if filenames:
                            file_metric["filename"] = Path(filenames[i]).name
                            per_file_results.append(file_metric)

                # --- 音声ファイルの保存 ---
                if output_wav_dir and filenames:
                    for i in range(enhanced.shape[0]):
                        save_file = output_path / Path(filenames[i]).name
                        torchaudio.save(str(save_file), enhanced[i].cpu().unsqueeze(0), self.sample_rate)

        # --- 結果の集計と保存 ---
        if output_wav_dir:
            print("Inference finished.")

        if calculate_metrics:
            if not per_file_results:
                print(
                    "Warning: No results to calculate metrics from. Ensure your dataloader provides filenames for CSV output.")
                return None

            df = pd.DataFrame(per_file_results)

            # データセット全体の平均を計算
            mean_metrics = df.mean(numeric_only=True)
            results = mean_metrics.to_dict()

            print("\nEvaluation finished. Average Results:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")

            # CSVファイルに保存
            if output_csv_dir:
                # 平均値の行を追加
                mean_row = mean_metrics.to_frame().T
                mean_row['filename'] = 'Average'
                df = pd.concat([df, mean_row], ignore_index=True)

                # 列の順序を調整
                cols = ['filename'] + [col for col in df.columns if col != 'filename']
                df = df[cols]

                csv_path = output_csv_path / "evaluation_results.csv"
                df.to_csv(csv_path, index=False, float_format='%.4f')
                print(f"\nPer-file and average results saved to {csv_path}")

            return results

        return None
    def _save_checkpoint(self, epoch: int, is_best: bool):
        """モデルのチェックポイントを保存する。"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        # 最新のチェックポイントを保存
        latest_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(state, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")

        # 最良モデルを保存
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(state, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: str):
        """指定されたパスからチェックポイントを読み込む。"""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found at {path}. Starting from scratch.")
            return

        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Checkpoint loaded. Resuming training from epoch {self.start_epoch}.")

    def _log_metrics(self, metrics: Dict[str, float], step: int, phase: str):
        """TensorBoardにメトリクスを記録する。"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
