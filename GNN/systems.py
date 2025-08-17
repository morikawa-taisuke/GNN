import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from itertools import permutations

# --- モデルのインポート ---
# プロジェクト内の様々なモデルをconfigに基づいて切り替えられるようにインポートします
from models.GCN import UGCNNet, UGATNet, UGCNNet2, UGATNet2
from models.SpeqGNN import SpeqGCNNet, SpeqGATNet, SpeqGCNNet2, SpeqGATNet2
from models.ConvTasNet_models import enhance_ConvTasNet
from models.wave_unet import U_Net


# --- 損失関数とヘルパー関数 ---
# main.pyから移植し、LightningModule内で直接利用できるようにします
def padding_tensor(tensor1, tensor2):
	"""
	最後の次元（時系列長）が異なる2つのテンソルに対して、
	短い方をゼロパディングして長さをそろえます。
	"""
	len1 = tensor1.size(-1)
	len2 = tensor2.size(-1)
	max_len = max(len1, len2)
	padded_tensor1 = nn.functional.pad(tensor1, [0, max_len - len1])
	padded_tensor2 = nn.functional.pad(tensor2, [0, max_len - len2])
	return padded_tensor1, padded_tensor2


def si_sdr_loss(ests, egs, eps=1e-8):
	"""
	SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) 損失を計算します。
	PyTorch Lightningの作法に合わせ、ests (estimated) と egs (example/ground truth) を引数に取ります。
	"""

	def l2norm(mat, keepdim=False):
		return torch.norm(mat, dim=-1, keepdim=keepdim)

	if ests.shape != egs.shape:
		raise RuntimeError(f"Dimension mismatch when calculating si-sdr, {ests.shape} vs {egs.shape}")

	ests_zm = ests - torch.mean(ests, dim=-1, keepdim=True)
	egs_zm = egs - torch.mean(egs, dim=-1, keepdim=True)
	t = torch.sum(ests_zm * egs_zm, dim=-1, keepdim=True) * egs_zm / (l2norm(egs_zm, keepdim=True) ** 2 + eps)

	loss = 20 * torch.log10(eps + l2norm(t) / (l2norm(ests_zm - t) + eps))
	return -torch.mean(loss)


class EnhancementSystem(pl.LightningModule):
	"""
	音声強調のためのLightningModule。
	モデルの定義、学習、検証、推論のロジックをすべて内包します。
	"""

	def __init__(self, config):
		super().__init__()
		# configを保存し、チェックポイントから復元できるようにする
		self.save_hyperparameters(config)
		self.config = config

		# モデルをconfigに基づいて構築
		self.model = self._build_model()

		# 損失関数をconfigに基づいて設定
		self.criterion = self._build_criterion()

		# STFT/ISTFT変換器をドメインに応じて準備
		if self.config["model"]["domain"] == "frequency":
			stft_params = self.config["stft_params"]
			self.stft = torchaudio.transforms.Spectrogram(
				n_fft=stft_params.get("n_fft", 512),
				hop_length=stft_params.get("hop_length", 256),
				win_length=stft_params.get("win_length", 512),
				window_fn=torch.hann_window,
				power=None,  # 複素スペクトログラムを出力
				return_complex=True
			)
			self.istft = torchaudio.transforms.InverseSpectrogram(
				n_fft=stft_params.get("n_fft", 512),
				hop_length=stft_params.get("hop_length", 256),
				win_length=stft_params.get("win_length", 512),
				window_fn=torch.hann_window
			)

	def _build_model(self):
		"""configファイルの情報に基づいてモデルをインスタンス化するヘルパー関数。"""
		model_name = self.config["model"]["name"]
		num_mic = self.config["model"].get("num_mic", 1)
		num_node = self.config["model"].get("num_node", 8)

		# モデル名と対応するクラスをマッピング
		model_map = {
			"GCN": UGCNNet, "UGCN": UGCNNet,
			"GAT": UGATNet, "UGAT": UGATNet,
			"GCN2": UGCNNet2, "UGCN2": UGCNNet2,
			"GAT2": UGATNet2, "UGAT2": UGATNet2,
			"SpeqGCN": SpeqGCNNet,
			"SpeqGAT": SpeqGATNet,
			"SpeqGCN2": SpeqGCNNet2,
			"SpeqGAT2": SpeqGATNet2,
			"ConvTasNet": enhance_ConvTasNet,
			"UNet": U_Net,
		}

		if model_name in model_map:
			# GNN系のモデルは共通の引数を取る
			if "GCN" in model_name or "GAT" in model_name or "Speq" in model_name:
				return model_map[model_name](n_channels=num_mic, n_classes=1, num_node=num_node)
			else:  # ConvTasNet, UNetなど
				return model_map[model_name]()
		else:
			raise ValueError(f"モデル '{model_name}' はサポートされていません。")

	def _build_criterion(self):
		"""configファイルの情報に基づいて損失関数を返すヘルパー関数。"""
		criterion_name = self.config["model"].get("criterion", "SISDR")
		if criterion_name.upper() == "SISDR":
			return si_sdr_loss
		elif criterion_name.upper() == "MSE":
			return nn.MSELoss()
		else:
			raise ValueError(f"損失関数 '{criterion_name}' はサポートされていません。")

	def forward(self, x, *args, **kwargs):
		"""モデルの順伝播。"""
		return self.model(x, *args, **kwargs)

	def _prepare_batch(self, batch):
		"""バッチからデータを抽出し、モデルのドメインに応じた入力とターゲットを準備する。"""
		noisy_waveform, clean_waveform = batch

		if self.config["model"]["domain"] == "frequency":
			# SpeqGNNは複素スペクトログラムと元の波形長も必要
			if "Speq" in self.config["model"]["name"]:
				noisy_spec_complex = self.stft(noisy_waveform.squeeze(1))
				model_input = (torch.abs(noisy_spec_complex).unsqueeze(1), noisy_spec_complex, noisy_waveform.shape[-1])
			else:  # 他の周波数領域モデルはマグニチュードのみを想定
				model_input = torch.abs(self.stft(noisy_waveform))

			target = torch.abs(self.stft(clean_waveform))
		else:  # 時間領域
			model_input = noisy_waveform
			target = clean_waveform

		return model_input, target, noisy_waveform, clean_waveform

	def training_step(self, batch, batch_idx):
		model_input, target, _, clean_waveform = self._prepare_batch(batch)

		# モデルのフォワードパス
		if "Speq" in self.config["model"]["name"]:
			# SpeqGNNは複数の引数を取る
			enhanced_output = self.model(model_input[0], model_input[1], model_input[2])
		else:
			enhanced_output = self.model(model_input)

		# 損失計算のために出力を整形
		if self.config["model"]["domain"] == "time":
			enhanced_output, target = padding_tensor(enhanced_output, target)

		loss = self.criterion(enhanced_output, target)
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		model_input, target, _, clean_waveform = self._prepare_batch(batch)

		if "Speq" in self.config["model"]["name"]:
			enhanced_output = self.model(model_input[0], model_input[1], model_input[2])
		else:
			enhanced_output = self.model(model_input)

		if self.config["model"]["domain"] == "time":
			enhanced_output, target = padding_tensor(enhanced_output, target)

		val_loss = self.criterion(enhanced_output, target)
		self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
		return val_loss

	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		# 推論時はファイル名もバッチに含まれることを想定
		noisy_waveform, filename = batch
		original_len = noisy_waveform.shape[-1]

		if self.config["model"]["domain"] == "frequency":
			noisy_spec_complex = self.stft(noisy_waveform.squeeze(1))

			if "Speq" in self.config["model"]["name"]:
				# SpeqGNNはマグニチュード、複素スペクトログラム、元の長さを入力
				model_input = (torch.abs(noisy_spec_complex).unsqueeze(1), noisy_spec_complex, original_len)
				enhanced_waveform = self.model(*model_input)
			else:
				# 他の周波数モデルはマグニチュードを入力とし、マグニチュードを出力
				model_input = torch.abs(noisy_spec_complex)
				enhanced_magnitude = self.model(model_input)
				# 元のノイズの位相を使って波形に復元
				reconstructed_spec = torch.polar(enhanced_magnitude, torch.angle(noisy_spec_complex))
				enhanced_waveform = self.istft(reconstructed_spec, length=original_len)
		else:  # 時間領域
			enhanced_waveform = self.model(noisy_waveform)

		return enhanced_waveform, filename

	def configure_optimizers(self):
		"""オプティマイザと学習率スケジューラを設定する。"""
		optim_config = self.config["optim"]
		optimizer = torch.optim.Adam(
			self.parameters(),
			lr=optim_config.get("lr", 0.001),
		)

		if optim_config.get("scheduler") == "reduce_on_plateau":
			scheduler_params = optim_config.get("scheduler_params", {})
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer,
				mode=scheduler_params.get("mode", "min"),
				factor=scheduler_params.get("factor", 0.5),
				patience=scheduler_params.get("patience", 5),
			)
			return {
				"optimizer": optimizer,
				"lr_scheduler": {
					"scheduler": scheduler,
					"monitor": "val_loss",  # 監視するメトリクス
				},
			}
		return optimizer