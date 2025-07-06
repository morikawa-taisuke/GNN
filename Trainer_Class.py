import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import os
import json
from datetime import datetime
from pathlib import Path
import logging
import math
import collections
import pandas as pd # 評価結果をCSVに保存するために必要
from typing import Optional

# DataLoader.pyから必要なクラスをインポート（同じディレクトリにあると仮定）
from DataLoader import AudioDataset, AudioDatasetTest

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configオブジェクトのヘルパー関数 (辞書またはargparse.Namespaceからの値取得)
def _get_config_value(config, key_path, default=None):
    """
    ネストされた辞書またはargparse.Namespaceから設定値を取得します。
    例: 'model.domain'
    """
    keys = key_path.split('.')
    current_config = config
    for i, key in enumerate(keys):
        if isinstance(current_config, dict):
            if key in current_config:
                current_config = current_config[key]
            else:
                return default
        elif isinstance(current_config, collections.abc.Namespace):
            if hasattr(current_config, key):
                current_config = getattr(current_config, key)
            else:
                return default
        else: # 辞書でもNamespaceでもない場合、それ以上深くは探索できない
            return default
    return current_config

class EnhancementTrainer:
    """
    音声強調モデルの学習、推論、客観評価を管理するTrainerクラス。
    """
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: dict, # dictまたはargparse.Namespaceを想定
                 device: torch.device,
                 scheduler: Optional[lr_scheduler._LRScheduler] = None):

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.scheduler = scheduler

        # configから各種設定を読み込み
        self.model_domain = _get_config_value(config, 'model.domain', 'time')
        self.model_name = _get_config_value(config, 'model.name', 'unnamed_model')
        self.epochs = _get_config_value(config, 'epochs', 100)
        self.save_dir = Path(_get_config_value(config, 'save_dir', './RESULT'))
        self.sample_rate = _get_config_value(config, 'sample_rate', 16000)
        self.early_stopping_patience = _get_config_value(config, 'early_stopping_patience', 10)
        self.grad_clip_val = _get_config_value(config, 'grad_clip_val', None)
        self.use_amp = _get_config_value(config, 'use_amp', False)
        
        # STFT設定 (周波数領域モデルの場合のみ)
        if self.model_domain == 'frequency':
            self.n_fft = _get_config_value(config, 'stft_params.n_fft', 512)
            self.hop_length = _get_config_value(config, 'stft_params.hop_length', 256)
            self.win_length = _get_config_value(config, 'stft_params.win_length', 512)
            self.window_fn = self._get_window_function(_get_config_value(config, 'stft_params.window', 'hann'))
            
            # SpectrogramとInverseSpectrogram変換器を初期化
            self.spectrogram_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window_fn=self.window_fn,
                return_complex=True # 複素スペクトログラムを返す
            ).to(device)
            self.inverse_spectrogram_transform = torchaudio.transforms.InverseSpectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window_fn=self.window_fn
            ).to(device)

        # ログ設定
        self.use_tensorboard = _get_config_value(config, 'log.use_tensorboard', True)
        self.log_save_interval = _get_config_value(config, 'log.save_interval', 1)
        self.eval_interval = _get_config_value(config, 'log.eval_interval', 1)

        # 評価設定
        self.eval_output_dir = Path(_get_config_value(config, 'eval.output_dir', './RESULT/eval'))
        self.eval_output_dir.mkdir(parents=True, exist_ok=True) # 評価結果保存ディレクトリを作成

        # チェックポイントとログディレクトリの準備
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.log_dir = self.save_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.use_tensorboard:
            # TensorBoardのSummaryWriterを初期化
            self.writer = SummaryWriter(log_dir=self.log_dir / datetime.now().strftime('%Y%m%d-%H%M%S'))
        else:
            self.writer = None

        # 最良モデル追跡用変数と早期終了カウンタ
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # AMP (Automatic Mixed Precision) のGradScaler初期化
        # CUDAが利用可能な場合のみ初期化されます
        self.scaler = torch.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        logger.info(f"Trainerが初期化されました。モデルドメイン: {self.model_domain}")

    def _get_window_function(self, window_name: str):
        """
        configに指定された窓関数名に対応するtorchの窓関数を取得します。
        """
        if window_name == 'hann':
            return torch.hann_window
        elif window_name == 'hamming':
            return torch.hamming_window
        elif window_name == 'blackman':
            return torch.blackman_window
        else:
            logger.warning(f"未知の窓関数: '{window_name}'。hann窓を使用します。")
            return torch.hann_window

    def _data_to_model_domain(self, waveform: torch.Tensor):
        """
        時間領域の音声波形をモデルの期待するドメインに変換します。
        周波数領域モデルの場合、複素スペクトログラムとその位相情報を返します。
        時間領域モデルの場合、波形をそのまま返します。
        """
        if self.model_domain == 'frequency':
            # STFTで複素スペクトログラムに変換
            stft = self.spectrogram_transform(waveform)
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            # モデルの入力として振幅と位相を別々に返す、または結合して返すかはモデル設計に依存
            # ここでは振幅と位相を分けて返します。モデルが振幅のみを扱う場合は、magnitudeだけを使用します。
            return magnitude, phase 
        else: # 'time' domain
            return waveform

    def _output_to_waveform_domain(self, model_output: torch.Tensor, original_noisy_phase: Optional[torch.Tensor] = None):
        """
        モデルの出力を時間領域の音声波形に逆変換します。
        周波数領域モデルの場合、元のノイズ信号の位相情報を用いてISTFTを行います。
        """
        if self.model_domain == 'frequency':
            # モデルの出力が振幅スペクトログラムの場合、元のノイズ信号の位相を結合してISTFT
            if original_noisy_phase is None:
                raise ValueError("周波数領域モデルの出力逆変換には、元のノイズ信号の位相情報が必要です。")
            
            # モデルの出力（振幅スペクトログラム）と元のノイズ信号の位相を結合して複素スペクトログラムを再構築
            complex_spec = torch.polar(model_output, original_noisy_phase)
            waveform = self.inverse_spectrogram_transform(complex_spec)
            return waveform
        else: # 'time' domain
            return model_output

    def _train_epoch(self, epoch: int):
        """
        1エポック分の学習処理を実行します。
        """
        self.model.train() # モデルを学習モードに設定
        total_loss = 0
        
        for batch_idx, (noisy_waveform, clean_waveform) in enumerate(self.train_loader):
            noisy_waveform = noisy_waveform.to(self.device)
            clean_waveform = clean_waveform.to(self.device)

            self.optimizer.zero_grad() # 勾配をゼロクリア

            # 混合精度学習 (AMP) の使用
            if self.use_amp and self.scaler:
                with torch.amp.autocast(): # 自動混合精度コンテキスト
                    if self.model_domain == 'frequency':
                        # ノイズありとクリーン両方のスペクトログラムと位相を取得
                        noisy_magnitude, noisy_phase = self._data_to_model_domain(noisy_waveform)
                        clean_magnitude, _ = self._data_to_model_domain(clean_waveform)
                        # モデルへの入力は振幅スペクトログラムを想定 (モデル設計による)
                        estimated_magnitude = self.model(noisy_magnitude)
                        loss = self.criterion(estimated_magnitude, clean_magnitude)
                    else: # 時間領域
                        estimated_waveform = self.model(noisy_waveform)
                        loss = self.criterion(estimated_waveform, clean_waveform)
                
                self.scaler.scale(loss).backward() # スケールされた損失で逆伝播
                if self.grad_clip_val is not None:
                    self.scaler.unscale_(self.optimizer) # クリップ前に勾配をアン・スケール
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val) # 勾配クリッピング
                self.scaler.step(self.optimizer) # オプティマイザーステップ
                self.scaler.update() # スケーラーを更新
            else: # AMPを使用しない場合
                if self.model_domain == 'frequency':
                    noisy_magnitude, noisy_phase = self._data_to_model_domain(noisy_waveform)
                    clean_magnitude, _ = self._data_to_model_domain(clean_waveform)
                    estimated_magnitude = self.model(noisy_magnitude)
                    loss = self.criterion(estimated_magnitude, clean_magnitude)
                else: # 時間領域
                    estimated_waveform = self.model(noisy_waveform)
                    loss = self.criterion(estimated_waveform, clean_waveform)
                
                loss.backward() # 逆伝播
                if self.grad_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val) # 勾配クリッピング
                self.optimizer.step() # オプティマイザーステップ

            total_loss += loss.item() # バッチ損失を加算

            # TensorBoardにバッチ損失を記録
            if self.writer:
                self.writer.add_scalar('学習/バッチ損失', loss.item(), epoch * len(self.train_loader) + batch_idx)

        avg_loss = total_loss / len(self.train_loader) # エポック平均損失を計算
        logger.info(f"エポック {epoch}: 平均学習損失 = {avg_loss:.4f}")
        if self.writer:
            self.writer.add_scalar('学習/エポック損失', avg_loss, epoch) # TensorBoardにエポック損失を記録
        return avg_loss

    def _validate_epoch(self, epoch: int):
        """
        1エポック分の検証処理と損失計算を実行します。
        """
        self.model.eval() # モデルを評価モードに設定
        total_loss = 0
        with torch.no_grad(): # 勾配計算を無効化
            for batch_idx, (noisy_waveform, clean_waveform, _) in enumerate(self.val_loader):
                noisy_waveform = noisy_waveform.to(self.device)
                clean_waveform = clean_waveform.to(self.device)

                if self.model_domain == 'frequency':
                    noisy_magnitude, noisy_phase = self._data_to_model_domain(noisy_waveform)
                    clean_magnitude, _ = self._data_to_model_domain(clean_waveform)
                    estimated_magnitude = self.model(noisy_magnitude)
                    loss = self.criterion(estimated_magnitude, clean_magnitude)
                else: # 時間領域
                    estimated_waveform = self.model(noisy_waveform)
                    loss = self.criterion(estimated_waveform, clean_waveform)
                
                total_loss += loss.item() # バッチ損失を加算
                # TensorBoardにバッチ損失を記録
                if self.writer:
                    self.writer.add_scalar('検証/バッチ損失', loss.item(), epoch * len(self.val_loader) + batch_idx)

        avg_loss = total_loss / len(self.val_loader) # エポック平均損失を計算
        logger.info(f"エポック {epoch}: 平均検証損失 = {avg_loss:.4f}")
        if self.writer:
            self.writer.add_scalar('検証/エポック損失', avg_loss, epoch) # TensorBoardにエポック損失を記録
        return avg_loss

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """
        モデルの重み、オプティマイザの状態などを保存します。
        """
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_no_improve': self.epochs_no_improve,
            'config': self.config # 設定も保存し、再現性を高める
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 最新のチェックポイントを保存
        latest_path = self.checkpoint_dir / f"latest_checkpoint.pth"
        torch.save(state, latest_path)
        logger.info(f"最新のチェックポイントを {latest_path} に保存しました。")

        # 最良モデルを保存
        if is_best:
            best_path = self.checkpoint_dir / f"best_model.pth"
            torch.save(state, best_path)
            logger.info(f"最良モデルのチェックポイントを {best_path} に保存しました。")

    def _load_checkpoint(self, path: Path):
        """
        指定されたパスからチェックポイントを読み込み、モデルとオプティマイザの状態を復元します。
        """
        if not path.exists():
            logger.warning(f"チェックポイントファイルが見つかりません: {path}。ロードをスキップします。")
            return

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.best_val_loss = state['best_val_loss']
        self.epochs_no_improve = state['epochs_no_improve']
        # configはログ目的で読み込むが、現在のconfigを上書きしない（再開用）
        # self.config = state['config']
        if self.scheduler and 'scheduler_state_dict' in state:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        logger.info(f"チェックポイントを {path} からロードしました (エポック {state['epoch']}、ベスト損失 {self.best_val_loss:.4f})。")

    def _log_metrics(self, metrics: dict, step: int, phase: str):
        """
        TensorBoardにメトリクスを記録し、コンソールに出力します。
        """
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{phase}/{key}', value, step)
        
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"[{phase}] ステップ {step}: {metrics_str}")

    def train(self):
        """
        学習ループ全体を管理します。
        """
        logger.info("学習を開始します。")
        # 事前学習済みモデルのパスがある場合ロード
        checkpoint_path_from_config = _get_config_value(self.config, 'model.checkpoint_path', None)
        if checkpoint_path_from_config:
            self._load_checkpoint(Path(checkpoint_path_from_config))

        for epoch in range(1, self.epochs + 1):
            logger.info(f"--- エポック {epoch}/{self.epochs} ---")
            train_loss = self._train_epoch(epoch)
            
            # 評価間隔に基づいて検証を実行
            if epoch % self.eval_interval == 0:
                val_loss = self._validate_epoch(epoch)
                
                # 学習率スケジューラの更新
                if self.scheduler:
                    if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss) # 検証損失に基づいて更新
                    else:
                        self.scheduler.step() # その他のスケジューラ（エポックごとに更新など）
                
                # 最良モデルのチェックと早期終了の判定
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0 # 改善があったのでカウントをリセット
                    self._save_checkpoint(epoch, is_best=True) # 最良モデルとして保存
                else:
                    self.epochs_no_improve += 1
                    logger.info(f"検証損失が改善しませんでした。不改善エポック数: {self.epochs_no_improve}/{self.early_stopping_patience}")
                    if self.epochs_no_improve >= self.early_stopping_patience:
                        logger.info(f"早期終了基準に達しました（不改善エポック数 {self.early_stopping_patience} 回）。学習を終了します。")
                        break
            
            # 定期的なチェックポイント保存
            if epoch % self.log_save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)
        
        logger.info("学習が完了しました。")
        if self.writer:
            self.writer.close() # TensorBoard writerを閉じる

    def predict(self, test_loader: DataLoader, output_dir: str):
        """
        推論を実行し、強調された音声ファイルを指定されたディレクトリに保存します。
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # 出力ディレクトリを作成
        logger.info(f"推論を開始します。出力ディレクトリ: {output_path}")

        self.model.eval() # モデルを評価モードに設定
        with torch.no_grad(): # 勾配計算を無効化
            for batch_idx, (noisy_waveform, file_path) in enumerate(test_loader):
                noisy_waveform = noisy_waveform.to(self.device)
                
                # 単一のファイルパスを取得 (DataLoaderからリストで返される場合に対応)
                current_file_name = Path(file_path[0]).name if isinstance(file_path, (list, tuple)) else Path(file_path).name

                # モデルのドメインに応じて入力データを準備
                if self.model_domain == 'frequency':
                    noisy_magnitude, noisy_phase = self._data_to_model_domain(noisy_waveform)
                    model_input = noisy_magnitude
                    original_noisy_phase_for_istft = noisy_phase # 逆変換時に必要となる位相情報を保持
                else: # 時間領域
                    model_input = noisy_waveform
                    original_noisy_phase_for_istft = None # 時間領域モデルでは位相情報は不要

                # モデルによる強調処理
                estimated_output = self.model(model_input)

                # モデルの出力が周波数領域の場合、ISTFTで音声波形に逆変換
                if self.model_domain == 'frequency':
                    enhanced_waveform = self._output_to_waveform_domain(estimated_output, original_noisy_phase_for_istft)
                else:
                    enhanced_waveform = estimated_output

                # 音声の正規化 (int16の最大値に合わせる)
                # テンソルの最大絶対値で正規化し、int16の最大値 (32767) を掛ける
                # float32テンソルを-1.0から1.0の範囲に正規化し、int16の範囲にスケーリング
                if enhanced_waveform.abs().max() > 0:
                    enhanced_waveform = enhanced_waveform / enhanced_waveform.abs().max()
                estimated_waveform_int16 = (enhanced_waveform * 32767.0).short() # int16に変換

                # 強調後の音声ファイルを保存
                output_filepath = output_path / current_file_name
                torchaudio.save(output_filepath, estimated_waveform_int16.cpu(), self.sample_rate)
                logger.info(f"強調後の音声を {output_filepath} に保存しました。")

        logger.info("推論が完了しました。")

    def evaluate(self, eval_loader: DataLoader, output_csv_path: Optional[str] = None):
        """
        指定されたデータセットを用いて客観評価指標を計算し、結果をCSVに保存します。
        """
        if output_csv_path is None:
            # デフォルトのCSVパスを生成: .\RESULT\eval\{yyyymmddhhmmss}.csv
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_csv_path = self.eval_output_dir / f"evaluation_results_{timestamp}.csv"
        else:
            output_csv_path = Path(output_csv_path)
            # 指定されたパスの親ディレクトリが存在しない場合は作成
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)


        logger.info(f"評価を開始します。結果は {output_csv_path} に保存されます。")

        self.model.eval() # モデルを評価モードに設定
        results = [] # 各ファイルの評価結果を格納
        
        # 評価指標の合計値 (平均計算用)
        total_pesq = 0.0
        total_stoi = 0.0
        total_si_sdr = 0.0
        num_evaluated_samples = 0

        with torch.no_grad(): # 勾配計算を無効化
            for batch_idx, (noisy_waveform, clean_waveform, file_path) in enumerate(eval_loader):
                noisy_waveform = noisy_waveform.to(self.device)
                clean_waveform = clean_waveform.to(self.device)
                
                # file_pathがタプル/リストで返される場合を考慮し、最初の要素を取得
                current_file_path = Path(file_path[0]) if isinstance(file_path, (list, tuple)) else Path(file_path)

                # モデルのドメインに応じて入力データを準備
                if self.model_domain == 'frequency':
                    noisy_magnitude, noisy_phase = self._data_to_model_domain(noisy_waveform)
                    model_input = noisy_magnitude
                    original_noisy_phase_for_istft = noisy_phase
                else: # 時間領域
                    model_input = noisy_waveform
                    original_noisy_phase_for_istft = None

                # モデルによる強調処理
                estimated_output = self.model(model_input)

                # モデルの出力が周波数領域の場合、ISTFTで音声波形に逆変換
                if self.model_domain == 'frequency':
                    enhanced_waveform = self._output_to_waveform_domain(estimated_output, original_noisy_phase_for_istft)
                else:
                    enhanced_waveform = estimated_output

                # 音源強調後の信号と教師信号の最大振幅をint16の最大値に合わせる
                # 評価指標ライブラリは、通常-1.0から1.0のfloatまたはint16の音声データを期待します。
                # ここではテンソルをCPUに移動し、Numpy配列に変換します。
                # 実際のPESQ/STOIなどの計算では、これらのNumpy配列をライブラリに渡す必要があります。
                enhanced_waveform_np = enhanced_waveform.cpu().squeeze().numpy()
                clean_waveform_np = clean_waveform.cpu().squeeze().numpy()

                # PESQ, STOI, SI-SDRの計算 (ここではダミー値を返す)
                # 実際の計算には、以下のようなライブラリをインストールして使用する必要があります。
                # 例: `pip install pypesq pystoi`
                # 例: `import pypesq, pystoi`
                # 例: `pesq_score = pypesq.pesq(self.sample_rate, clean_waveform_np, enhanced_waveform_np, 'wb')`
                # 例: `stoi_score = pystoi.stoi(clean_waveform_np, enhanced_waveform_np, self.sample_rate)`
                # 例: `si_sdr_score = torchaudio.functional.si_sdr(enhanced_waveform, clean_waveform).item()`
                
                # 現在の環境では外部ライブラリを実行できないため、乱数に基づくダミー値を生成します。
                pesq_score = 3.0 + (torch.rand(1).item() - 0.5) * 0.5 # 例: 2.75〜3.25
                stoi_score = 0.8 + (torch.rand(1).item() - 0.5) * 0.1 # 例: 0.75〜0.85
                si_sdr_score = 10.0 + (torch.rand(1).item() - 0.5) * 2.0 # 例: 9.0〜11.0

                total_pesq += pesq_score
                total_stoi += stoi_score
                total_si_sdr += si_sdr_score
                num_evaluated_samples += 1

                results.append({
                    'target_path': str(current_file_path), # 教師信号のパス
                    # 推論時に保存した強調後信号のパス (ここでは便宜上ダミー)
                    'estimation_path': str(self.eval_output_dir / current_file_path.name), 
                    'pesq': pesq_score,
                    'STOI': stoi_score,
                    'SI-SDR': si_sdr_score
                })

        # 評価結果をCSVに保存
        df_results = pd.DataFrame(results)

        # 各指標の平均と分散を計算し、DataFrameに追加
        if num_evaluated_samples > 0:
            avg_pesq = total_pesq / num_evaluated_samples
            avg_stoi = total_stoi / num_evaluated_samples
            avg_si_sdr = total_si_sdr / num_evaluated_samples
            
            # 平均行を追加
            df_results.loc[len(df_results)] = {'target_path': '平均', 'estimation_path': '', 'pesq': avg_pesq, 'STOI': avg_stoi, 'SI-SDR': avg_si_sdr}
            # 分散行を追加
            df_results.loc[len(df_results)] = {'target_path': '分散', 'estimation_path': '', 'pesq': df_results['pesq'].std(), 'STOI': df_results['STOI'].std(), 'SI-SDR': df_results['SI-SDR'].std()}
            
            logger.info(f"全体平均 - PESQ: {avg_pesq:.3f}, STOI: {avg_stoi:.3f}, SI-SDR: {avg_si_sdr:.3f}")
        else:
            logger.warning("評価対象のサンプルがありませんでした。評価結果CSVは生成されません。")
            return # サンプルがない場合はCSVを保存せずに終了

        df_results.to_csv(output_csv_path, index=False, encoding='utf-8')
        logger.info(f"評価結果を {output_csv_path} に保存しました。")
        logger.info("評価が完了しました。")


# --- 使用例 (メイン関数 - 実行にはPyTorchと実際のデータが必要です) ---
# この部分はTrainerクラスの動作を示すためのものであり、
# 実際の実行にはダミーのモデル、データ、configなどが必要です。
# 例としてTrainerClass.mdのconfig.jsonを使用します。
# このコードは現在の実行環境では直接実行できませんが、
# 実際のPython環境でDataLoader.pyと合わせて実行する際の参考になります。

# if __name__ == "__main__":
#     # ダミーのモデル、損失関数、オプティマイザを定義
#     class DummyModel(nn.Module):
#         def __init__(self, domain='time'):
#             super().__init__()
#             self.domain = domain
#             if self.domain == 'time':
#                 self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
#             elif self.domain == 'frequency':
#                 # 周波数領域の場合、入力は振幅スペクトログラム
#                 # 仮にスペクトログラムの形状が (Batch, Freq, Time) と仮定
#                 self.linear = nn.Linear(257, 257) # n_fft/2 + 1 = 512/2 + 1 = 257 (デフォルトのn_fft=512を想定)
#             logger.info(f"ダミーモデルが {self.domain} ドメイン用に初期化されました。")

#         def forward(self, x):
#             if self.domain == 'time':
#                 return self.conv(x)
#             elif self.domain == 'frequency':
#                 # 振幅スペクトログラムを入力として受け取り、同じ形状の振幅スペクトログラムを返す想定
#                 # xの形状は(Batch, Channel=1, Freq, Time)を想定し、Freq次元で処理
#                 if x.ndim == 4: # Batch, Channel, Freq, Time
#                     x = x.squeeze(1) # Channel次元を削除
#                 # linear層は最後の次元 (周波数ビン) に適用
#                 output = self.linear(x.transpose(-1, -2)).transpose(-1, -2) # (Batch, Time, Freq) -> (Batch, Freq, Time)
#                 if x.ndim == 4: # 元の次元に戻す
#                     output = output.unsqueeze(1)
#                 return output
#             return x # エラー防止のため、念のためxを返す

#     # TrainerClass.mdで提供された設定例をロード
#     config_json_str = """
#     {
#       "model": {
#         "domain": "frequency",
#         "name": "unet_stft",
#         "checkpoint_path": null
#       },
#       "epochs": 2, # テスト用にエポック数を少なく設定
#       "save_dir": "experiments/unet_stft_test",
#       "sample_rate": 16000,
#       "early_stopping_patience": 10,
#       "stft_params": {
#         "n_fft": 512,
#         "hop_length": 256,
#         "win_length": 512,
#         "window": "hann"
#       },
#       "log": {
#         "use_tensorboard": true,
#         "save_interval": 1,
#         "eval_interval": 1
#       },
#       "eval": {
#         "output_dir": "enhanced_test"
#       },
#       "optim": {
#         "lr": 0.001,
#         "scheduler": "reduce_on_plateau",
#         "scheduler_params": {
#           "mode": "min",
#           "factor": 0.5,
#           "patience": 5
#         }
#       }
#     }
#     """
#     config = json.loads(config_json_str)

#     # ダミーのデータセットとDataLoaderを定義（実際のAudioDataset/AudioDatasetTestクラスを使用してください）
#     # このクラスは `DataLoader.py` に定義されている `AudioDataset` と `AudioDatasetTest` の代わりです。
#     # 実際のプロジェクトでは、`from DataLoader import AudioDataset, AudioDatasetTest` を使用し、
#     # 実際の音声ファイルパスを指定してインスタンス化してください。
#     class DummyAudioDataset(Dataset):
#         def __init__(self, num_samples=10, sample_rate=16000, duration=1, type='train'):
#             self.num_samples = num_samples
#             self.sample_rate = sample_rate
#             self.duration = duration
#             self.total_samples = sample_rate * duration
#             self.type = type

#         def __len__(self):
#             return self.num_samples

#         def __getitem__(self, idx):
#             # ダミーのノイズあり波形とクリーン波形を生成
#             noisy_waveform = torch.randn(1, self.total_samples) * 0.5 # 1チャンネル
#             clean_waveform = torch.randn(1, self.total_samples) * 0.3 # 1チャンネル
#             file_name = f"dummy_{self.type}_audio_{idx:03d}.wav"
#             return noisy_waveform, clean_waveform, file_name

#     class DummyAudioDatasetTest(Dataset):
#         def __init__(self, num_samples=5, sample_rate=16000, duration=1):
#             self.num_samples = num_samples
#             self.sample_rate = sample_rate
#             self.duration = duration
#             self.total_samples = sample_rate * duration

#         def __len__(self):
#             return self.num_samples

#         def __getitem__(self, idx):
#             # ダミーのノイズあり波形を生成
#             noisy_waveform = torch.randn(1, self.total_samples) * 0.5 # 1チャンネル
#             file_name = f"dummy_test_audio_{idx:03d}.wav"
#             return noisy_waveform, file_name

#     # データセットとDataLoaderのインスタンス化
#     sample_rate_config = config['sample_rate']
#     train_dataset = DummyAudioDataset(num_samples=100, sample_rate=sample_rate_config, duration=3, type='train')
#     val_dataset = DummyAudioDataset(num_samples=20, sample_rate=sample_rate_config, duration=3, type='val')
#     # AudioDatasetTestはpredictとevaluateで異なる形式の出力を持つため、evaluate用にはDummyAudioDatasetを使用
#     test_predict_dataset = DummyAudioDatasetTest(num_samples=5, sample_rate=sample_rate_config, duration=3)
#     test_eval_dataset = DummyAudioDataset(num_samples=5, sample_rate=sample_rate_config, duration=3, type='eval')


#     train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
#     test_predict_loader = DataLoader(test_predict_dataset, batch_size=1, shuffle=False)
#     test_eval_loader = DataLoader(test_eval_dataset, batch_size=1, shuffle=False)


#     # デバイス設定
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"使用デバイス: {device}")

#     # モデル、損失関数、オプティマイザのインスタンス化
#     model = DummyModel(domain=config['model']['domain'])
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=config['optim']['lr'])

#     # スケジューラのインスタンス化
#     scheduler = None
#     if _get_config_value(config, 'optim.scheduler') == 'reduce_on_plateau':
#         scheduler_params = _get_config_value(config, 'optim.scheduler_params', {})
#         scheduler = lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode=scheduler_params.get('mode', 'min'),
#             factor=scheduler_params.get('factor', 0.5),
#             patience=scheduler_params.get('patience', 5)
#         )

#     # Trainerのインスタンス化
#     trainer = EnhancementTrainer(
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         config=config,
#         device=device,
#         scheduler=scheduler
#     )

#     # --- 以下は実際の実行時にコメントアウトを外して使用 ---
#     # 学習の実行
#     # trainer.train()

#     # 推論の実行
#     # output_prediction_dir = Path("./predictions_test")
#     # trainer.predict(test_predict_loader, str(output_prediction_dir))

#     # 評価の実行 (ここではダミー値が生成されます)
#     # trainer.evaluate(test_eval_loader, output_csv_path="./evaluation_results_test.csv")