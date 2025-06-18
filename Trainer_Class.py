# trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable # 未使用のようですが、元のコードに合わせて残します
from itertools import permutations

import numpy as np
import time
import json
from pathlib import Path
from tqdm.contrib import tenumerate
from tqdm import tqdm
import os # os.environ のために必要

# 既存のプロジェクトモジュールからのインポート
import UGNNNet_DatasetClass # データセットクラス
from mymodule import my_func # ユーティリティ関数
# 評価関数
from evaluation.PESQ import pesq_evaluation
from evaluation.STOI import stoi_evaluation
from evaluation.SI_SDR import sisdr_evaluation

# CUDAのメモリ管理設定 (main_GCN_similarity_node.py から移動しても良いが、グローバルな設定なので注意)
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # 必要に応じて有効化

def padding_tensor(tensor1, tensor2):
    """
    最後の次元（例: 時系列長）が異なる2つのテンソルに対して、
    短い方を末尾にゼロパディングして長さをそろえる。
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
        # ターゲットと推定のチャンネル数が異なる場合があるため、エラーではなく警告に留めるか、
        # もしくは呼び出し側で適切に処理することを期待する。
        # ここでは元のロジックを維持し、エラーを発生させる。
        # しかし、一般的には1ch対1chの比較を想定している。
        # print(f"Warning: Dimension mismatch in sisdr: x.shape={x.shape}, s.shape={s.shape}")
        # 最小のチャンネル数に合わせるか、エラーとするかは設計次第
        raise RuntimeError(f"Dimension mismatch when calculate si-sdr, {x.shape} vs {s.shape}")

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (torch.sum(s_zm * s_zm, dim=-1, keepdim=True) + eps) # eps追加
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)) # x_zm - t に修正

def si_sdr_loss(ests, egs):
    # ests: estimation (B, C, T) or (B, T)
    # egs: target (B, C, T) or (B, T)
    
    # Ensure inputs are at least 2D (batch, time)
    if ests.ndim == 1: ests = ests.unsqueeze(0)
    if egs.ndim == 1: egs = egs.unsqueeze(0)
    # If channel dimension is missing (B, T), add it (B, 1, T)
    if ests.ndim == 2: ests = ests.unsqueeze(1)
    if egs.ndim == 2: egs = egs.unsqueeze(1)

    # Now ests and egs are (B, C, T)
    # For single-speaker (or single target source) scenario, C=1
    # If C > 1, it implies multi-speaker separation.
    # The original code seems to assume C is the number of speakers/sources.

    num_sources = ests.size(1) # Number of estimated sources
    
    if num_sources != egs.size(1):
        # This case needs careful handling. For now, assume they match or egs has 1 source to compare against all ests.
        # Or, if ests has 1 source and egs has multiple (e.g. music separation), pick the relevant one.
        # For simplicity, if ests is single source and egs is multi-source, we might compare ests to each in egs.
        # However, the permutation logic below assumes ests and egs have the same number of "speakers".
        # If it's a single source enhancement (C=1 for both), permutations are trivial.
        # print(f"Warning: Number of estimated sources ({ests.size(1)}) and target sources ({egs.size(1)}) mismatch.")
        # Fallback: if ests is (B,1,T) and egs is (B,1,T), then num_sources = 1
        pass


    if num_sources == 1: # Single source case, no permutation needed
        sdr = sisdr(ests.squeeze(1), egs.squeeze(1)) # Remove channel dim for sisdr function if it expects (B,T)
        return -torch.mean(sdr)

    # Multi-speaker / multi-source case with permutation
    batch_size = ests.size(0)
    all_sisdr_sums = torch.zeros(batch_size, device=ests.device)

    for i in range(batch_size): # Process each item in the batch
        # ests_i: (C, T), egs_i: (C, T)
        ests_i = ests[i]
        egs_i = egs[i]
        
        perms = list(permutations(range(num_sources)))
        sisdr_perm_max = -float('inf')

        for p in perms:
            current_sisdr_sum = 0
            for src_idx, target_idx in enumerate(p):
                # sisdr expects (N, S) where N is batch (or single item), S is samples
                # Here, we pass (1, T) effectively by unsqueezing if sisdr expects a batch dim
                # Or, if sisdr can handle (T) directly, then ests_i[src_idx] is fine.
                # Assuming sisdr can handle (T) or (1,T)
                sdr_val = sisdr(ests_i[src_idx].unsqueeze(0), egs_i[target_idx].unsqueeze(0))
                current_sisdr_sum += sdr_val.squeeze() # .squeeze() if sisdr returns (1,)
            
            if current_sisdr_sum > sisdr_perm_max:
                sisdr_perm_max = current_sisdr_sum
        
        all_sisdr_sums[i] = sisdr_perm_max / num_sources # Average SI-SDR for the best permutation

    return -torch.mean(all_sisdr_sums)


class ModelPipeline:
    def __init__(self, model, config_path="config.json"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Pipeline using device: {self.device}")
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"エラー: 設定ファイルが見つかりません: {config_path}")
            raise
        except json.JSONDecodeError:
            print(f"エラー: 設定ファイル ({config_path}) のJSON形式が正しくありません。")
            raise

        self.model = model.to(self.device)

        self.paths_config = self.config.get('paths', {})
        self.training_params_config = self.config.get('training_params', {})
        self.evaluation_params_config = self.config.get('evaluation_params', {})
        # self.model_params_config = self.config.get('model_params', {}) # Already used in main to init model

        # Ensure output directories exist
        my_func.make_dir(self.paths_config.get('model_output_dir', './results/pth_models'))
        my_func.make_dir(self.paths_config.get('inference_output_wav', './results/output_wavs'))
        eval_csv_path = self.paths_config.get('evaluation_output_csv', './results/evaluation_scores.csv')
        my_func.make_dir(Path(eval_csv_path).parent)
        my_func.make_dir(self.paths_config.get('log_dir', './logs'))

    def train_model(self, checkpoint_path: str = None):
        model_out_dir = Path(self.paths_config['model_output_dir'])
        model_base_name = self.paths_config['model_base_name']
        
        log_dir_base = Path(self.paths_config['log_dir'])
        writer_tensorboard_dir = log_dir_base / model_base_name
        my_func.make_dir(writer_tensorboard_dir)
        writer = SummaryWriter(log_dir=str(writer_tensorboard_dir))

        now = my_func.get_now_time()
        csv_log_path = writer_tensorboard_dir / f"{model_base_name}_{now}.csv"
        my_func.make_dir(csv_log_path.parent)

        clean_path = self.paths_config['train_clean']
        noisy_path = self.paths_config['train_noisy']
        loss_func_name = self.training_params_config['loss_function']
        batchsize = self.training_params_config['batch_size']
        train_count = self.training_params_config['epochs']
        earlystopping_threshold = self.training_params_config['early_stopping_threshold']
        learning_rate = self.training_params_config['learning_rate']
        checkpoint_save_suffix = self.training_params_config.get('checkpoint_save_suffix', '_checkpoint')


        with open(csv_log_path, "w") as csv_file:
            csv_file.write(f"dataset_noisy,model_identifier,loss_func\n{noisy_path},{model_out_dir / model_base_name},{loss_func_name}\n")
            csv_file.write("epoch,avg_loss\n") # Changed from model_loss_sum to avg_loss for clarity

        best_loss = np.inf
        earlystopping_count = 0

        dataset = UGNNNet_DatasetClass.AudioDataset(clean_path, noisy_path, 
                                                   sample_rate=self.evaluation_params_config.get('sample_rate', 16000)) # Pass SR
        dataset_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=4 if self.device.type == 'cuda' else 0)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_function_instance = None
        if loss_func_name == "wave_MSE" or loss_func_name == "stft_MSE":
            loss_function_instance = nn.MSELoss().to(self.device)

        start_epoch = 1
        checkpoint_to_load = checkpoint_path
        if checkpoint_to_load is None:
            default_checkpoint_path = model_out_dir / f"{model_base_name}{checkpoint_save_suffix}.pth"
            if default_checkpoint_path.exists():
                checkpoint_to_load = str(default_checkpoint_path)

        if checkpoint_to_load is not None and Path(checkpoint_to_load).exists():
            print(f"チェックポイントから学習を再開: {checkpoint_to_load}")
            checkpoint = torch.load(checkpoint_to_load, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("loss", np.inf) 
            print(f"エポック {start_epoch-1} から再開。前回の損失: {best_loss:.6f}")
        else:
            print("最初から学習を開始します。")

        print("====================")
        print(f"デバイス: {self.device}")
        print(f"モデル出力ディレクトリ: {model_out_dir}")
        print(f"モデルベース名: {model_base_name}")
        print(f"ノイズありデータセットパス: {noisy_path}")
        print(f"損失関数: {loss_func_name}")
        print(f"バッチサイズ: {batchsize}, エポック数: {train_count}, 学習率: {learning_rate}")
        print("====================")

        self.model.train()
        start_time = time.time()

        for epoch in range(start_epoch, train_count + 1):
            epoch_loss_sum = 0.0
            print(f"学習エポック: {epoch}")
            for _, (mix_data, target_data) in tenumerate(dataset_loader, total=len(dataset_loader)):
                mix_data, target_data = mix_data.to(self.device, non_blocking=True), target_data.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                
                # Ensure data is float32, original script had this, dataset might already do it
                mix_data = mix_data.to(torch.float32)
                target_data = target_data.to(torch.float32)
                
                estimate_data = self.model(mix_data)
                estimate_data, target_data = padding_tensor(estimate_data, target_data)

                current_loss = 0
                if loss_func_name == "SISDR":
                    current_loss = si_sdr_loss(estimate_data, target_data)
                elif loss_func_name == "wave_MSE":
                    current_loss = loss_function_instance(estimate_data, target_data)
                elif loss_func_name == "stft_MSE":
                    # Assuming estimate_data and target_data are (B, C, T)
                    # For STFT, usually process each channel independently or average if mono
                    # If C=1, squeeze(1) is fine. If C > 1, need to decide how to handle.
                    # For now, assume C=1 as in original code's stft part.
                    stft_estimate_data = torch.stft(estimate_data.squeeze(1), n_fft=1024, return_complex=True)
                    stft_target_data = torch.stft(target_data.squeeze(1), n_fft=1024, return_complex=True)
                    current_loss = loss_function_instance(torch.abs(stft_estimate_data), torch.abs(stft_target_data))
                
                epoch_loss_sum += current_loss.item()
                current_loss.backward()
                optimizer.step()
                
                del mix_data, target_data, estimate_data, current_loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            avg_epoch_loss = epoch_loss_sum / len(dataset_loader)
            
            checkpoint_save_path = model_out_dir / f"{model_base_name}{checkpoint_save_suffix}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss
            }, checkpoint_save_path)

            writer.add_scalar(f"{model_base_name}/loss", avg_epoch_loss, epoch)
            print(f"[{epoch}] 平均エポック損失: {avg_epoch_loss:.6f}")

            with open(csv_log_path, "a") as out_file:
                out_file.write(f"{epoch},{avg_epoch_loss}\n")

            if avg_epoch_loss < best_loss:
                print(f"{epoch:3} [エポック] | 新しい最良損失: {avg_epoch_loss:.6f} <- {best_loss:.6f}")
                best_model_path = model_out_dir / f"BEST_{model_base_name}.pth"
                torch.save(self.model.state_dict(), best_model_path)
                best_loss = avg_epoch_loss
                earlystopping_count = 0
            else:
                earlystopping_count += 1
                print(f"{epoch:3} [エポック] | 損失改善なし: {avg_epoch_loss:.6f} (最良: {best_loss:.6f}). カウント: {earlystopping_count}")
                if epoch > 10 and earlystopping_count >= earlystopping_threshold:
                    print(f"エポック {epoch} で早期終了します。")
                    break
            
            if epoch == 100: # 100エポック時点のモデルを保存
                epoch_100_model_path = model_out_dir / f"{model_base_name}_epoch100.pth"
                torch.save(self.model.state_dict(), epoch_100_model_path)
                print(f"エポック 100 のモデルを保存: {epoch_100_model_path}")

        final_model_path = model_out_dir / f"{model_base_name}_epoch{epoch}.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"最終モデルを保存: {final_model_path}")
        writer.close()

        time_end = time.time()
        time_sec = time_end - start_time
        time_h = float(time_sec) / 3600.0
        print(f"学習完了。所要時間: {time_h:.2f}時間")

    def infer_model(self):
        mix_dir = self.paths_config['inference_input_mix']
        output_wav_dir = Path(self.paths_config['inference_output_wav'])
        my_func.make_dir(output_wav_dir)

        model_load_path = Path(self.paths_config['model_output_dir']) / f"BEST_{self.paths_config['model_base_name']}.pth"
        
        if not model_load_path.exists():
            print(f"エラー: モデルファイルが見つかりません: {model_load_path}。学習を実行するか、設定を確認してください。")
            return

        print(f"推論用のモデルをロード中: {model_load_path}")
        # モデルのパラメータをconfigから読み込む必要がある場合、ここで再インスタンス化する
        # self.model = TheModelClass(**self.model_params_config).to(self.device) # 例
        self.model.load_state_dict(torch.load(model_load_path, map_location=self.device))
        self.model.eval()

        filelist_mixdown = my_func.get_file_list(mix_dir, ext=".wav") # Ensure only wav files
        print(f'推論対象の混合ファイル数: {len(filelist_mixdown)}')

        for fmixdown in tqdm(filelist_mixdown, desc="推論処理中"):
            mix, prm = my_func.load_wav(str(fmixdown)) # my_func.load_wavがPathオブジェクトを扱えるか確認
            mix_tensor = torch.from_numpy(mix).float().to(self.device)
            
            original_ndim = mix_tensor.ndim
            if original_ndim == 1: # Mono: (length)
                mix_tensor = mix_tensor.unsqueeze(0) # (1, length) for channel
            # Now mix_tensor is (channels, length)
            mix_tensor = mix_tensor.unsqueeze(0) # (1, channels, length) for batch
            
            mix_abs_max = torch.max(torch.abs(mix_tensor)) # 元の振幅スケールを保持

            with torch.no_grad():
                estimate_tensor = self.model(mix_tensor) # (1, out_channels, length)
            
            estimate_tensor = estimate_tensor.squeeze(0) # (out_channels, length)
            
            # 推定信号の振幅を元の信号の最大絶対値にスケーリング（任意）
            if mix_abs_max > 1e-8: # ゼロ除算を避ける
                current_max_abs = torch.max(torch.abs(estimate_tensor))
                if current_max_abs > 1e-8:
                    estimate_tensor = estimate_tensor * (mix_abs_max / current_max_abs)
            
            separated_audio = estimate_tensor.cpu().numpy() # (out_channels, length)
            
            # save_wavが (length,) or (2, length) を期待する場合
            if separated_audio.shape[0] == 1 and original_ndim == 1: # Input was mono, output is (1, L)
                separated_audio = separated_audio.squeeze(0) # (L,)
            elif separated_audio.shape[0] > 1 and original_ndim == 1: # Input was mono, output is (C, L) C>1
                # この場合、どう保存するかは要件による。ここでは最初のチャンネルのみ保存する例
                print(f"警告: モノラル入力に対し複数チャンネルが出力されました。最初のチャンネルを保存します。 Shape: {separated_audio.shape}")
                separated_audio = separated_audio[0]


            foutname_stem = Path(fmixdown).stem
            output_filepath = output_wav_dir / f"{foutname_stem}_enhanced.wav"
            
            sr_to_save = prm.get('framerate') if prm and 'framerate' in prm else self.evaluation_params_config.get('sample_rate', 16000)
            
            # my_func.save_wav の仕様を確認: prm_stereo は bool か、チャンネル数か
            # ここでは separated_audio が (L,) ならモノラル、(C, L) C>1 ならステレオと仮定
            is_stereo_output = separated_audio.ndim > 1 and separated_audio.shape[0] > 1
            my_func.save_wav(str(output_filepath), separated_audio, sr_to_save, prm_stereo=is_stereo_output)
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        print(f"推論完了。強調処理されたファイルは {output_wav_dir} に保存されました。")

    def evaluate_model(self):
        target_dir = self.paths_config['evaluation_target']
        estimation_dir = self.paths_config['inference_output_wav']
        eval_out_csv_path = Path(self.paths_config['evaluation_output_csv'])
        sample_rate = self.evaluation_params_config.get('sample_rate', 16000)

        print(f"評価を開始します...")
        print(f"ターゲット（クリーン）ディレクトリ: {target_dir}")
        print(f"推定（強調処理後）ディレクトリ: {estimation_dir}")
        print(f"評価結果CSV: {eval_out_csv_path}")

        my_func.make_dir(eval_out_csv_path.parent)
        with open(eval_out_csv_path, "w") as csv_file:
            csv_file.write(f"target_dir,{target_dir}\n")
            csv_file.write(f"estimation_dir,{estimation_dir}\n")
            csv_file.write(f"sample_rate,{sample_rate}\n")
            csv_file.write("target_name,estimation_name,pesq,stoi,sisdr\n")

        target_list = my_func.get_file_list(dir_path=target_dir, ext=".wav")
        
        if not target_list:
            print(f"{target_dir} にターゲットファイルが見つかりませんでした。評価をスキップします。")
            return

        all_pesq_scores = []
        all_stoi_scores = []
        all_sisdr_scores = []

        for target_file_path_str in tqdm(target_list, desc="評価中"):
            target_file_path = Path(target_file_path_str)
            target_name_stem = target_file_path.stem
            
            estimation_file_path = Path(estimation_dir) / f"{target_name_stem}_enhanced.wav" 
            if not estimation_file_path.exists():
                estimation_file_path_alt = Path(estimation_dir) / f"{target_name_stem}.wav"
                if estimation_file_path_alt.exists():
                    estimation_file_path = estimation_file_path_alt
                else:
                    print(f"警告: {target_file_path.name} に対応する推定ファイルが {estimation_dir} に見つかりません。スキップします。")
                    continue
            
            estimation_name = estimation_file_path.name

            target_data, target_prm = my_func.load_wav(str(target_file_path))
            estimation_data, est_prm = my_func.load_wav(str(estimation_file_path))

            sr_target = target_prm.get('framerate', sample_rate) if target_prm else sample_rate
            sr_est = est_prm.get('framerate', sample_rate) if est_prm else sample_rate
            
            if sr_target != sample_rate:
                print(f"警告: ターゲット {target_file_path.name} のSR ({sr_target}Hz) が設定 ({sample_rate}Hz) と異なります。リサンプルは行いません。評価指標のSR引数に注意してください。")
            if sr_est != sample_rate:
                print(f"警告: 推定 {estimation_file_path.name} のSR ({sr_est}Hz) が設定 ({sample_rate}Hz) と異なります。リサンプルは行いません。評価指標のSR引数に注意してください。")

            # Ensure mono for PESQ/STOI if they expect mono
            if target_data.ndim > 1: target_data = target_data[0] # Assuming first channel if stereo
            if estimation_data.ndim > 1: estimation_data = estimation_data[0]

            max_length = max(len(target_data), len(estimation_data))
            target_data_padded = np.pad(target_data, (0, max_length - len(target_data)), "constant")
            estimation_data_padded = np.pad(estimation_data, (0, max_length - len(estimation_data)), "constant")

            target_data_clean = np.nan_to_num(target_data_padded)
            estimation_data_clean = np.nan_to_num(estimation_data_padded)

            pesq_score, stoi_score, sisdr_score = np.nan, np.nan, np.nan

            if np.sum(np.abs(target_data_clean)) > 1e-6 and np.sum(np.abs(estimation_data_clean)) > 1e-6:
                try:
                    pesq_score = pesq_evaluation(target_data_clean, estimation_data_clean, sr=sample_rate)
                except Exception as e:
                    print(f"PESQ計算エラー ({target_name_stem}): {e}")
                try:
                    stoi_score = stoi_evaluation(target_data_clean, estimation_data_clean, sr=sample_rate)
                except Exception as e:
                    print(f"STOI計算エラー ({target_name_stem}): {e}")
            else:
                print(f"警告: {target_name_stem} のターゲットまたは推定音声が無音またはほぼ無音です。PESQ/STOI は NaN になります。")

            try:
                # sisdr_evaluation might expect specific shapes or normalization
                sisdr_score = sisdr_evaluation(target_data_clean, estimation_data_clean) # Assuming this function handles its inputs
            except Exception as e:
                print(f"SI-SDR計算エラー ({target_name_stem}): {e}")


            if not np.isnan(pesq_score): all_pesq_scores.append(pesq_score)
            if not np.isnan(stoi_score): all_stoi_scores.append(stoi_score)
            if not np.isnan(sisdr_score): all_sisdr_scores.append(sisdr_score)

            with open(eval_out_csv_path, "a") as csv_file:
                csv_file.write(f"{target_file_path.name},{estimation_name},{pesq_score:.4f},{stoi_score:.4f},{sisdr_score:.4f}\n")

        if all_pesq_scores or all_stoi_scores or all_sisdr_scores:
            pesq_ave = np.mean(all_pesq_scores) if all_pesq_scores else np.nan
            stoi_ave = np.mean(all_stoi_scores) if all_stoi_scores else np.nan
            sisdr_ave = np.mean(all_sisdr_scores) if all_sisdr_scores else np.nan
            
            with open(eval_out_csv_path, "a") as csv_file:
                csv_file.write(f"average,,{pesq_ave:.4f},{stoi_ave:.4f},{sisdr_ave:.4f}\n")
            print(f"平均 PESQ: {pesq_ave:.4f}")
            print(f"平均 STOI: {stoi_ave:.4f}")
            print(f"平均 SI-SDR: {sisdr_ave:.4f}")
        else:
            print("評価可能なファイルペアがありませんでした。")
        print("評価が完了しました。")
