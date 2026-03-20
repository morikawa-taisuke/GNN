#
# mymodule/GraphLearningLoss.py
#
import torch
import torch.nn as nn
from typing import Tuple

# 既存のLossFunction.pyからget_loss_computerをインポート
from mymodule.LossFunction import get_loss_computer


# Document/アイデア2_E2Eのグラフ構造学習.md に基づく
# L_total = L_main + β * L_reg を計算するクラス

class GraphLearningLoss(nn.Module):
	"""
	動的グラフ学習のための総損失（L_total）を計算するモジュール。

	L_total = L_main + β * L_reg
	"""

	def __init__(self,
	             main_loss_name: str,
	             graph_reg_beta: float,
	             device: torch.device):
		"""
		Args:
			main_loss_name (str):
				主損失の名前 ("SISDR", "wave_MSE", "stft_MSE" など)

			graph_reg_beta (float):
				グラフ正則化損失の重み (β)

			device (torch.device):
				損失計算に使用するデバイス
		"""
		super().__init__()

		# 1. 主損失 (L_main) のためのコンピュータを取得
		self.main_loss_computer = get_loss_computer(main_loss_name, device)

		# 2. グラフ正則化の重み (β) を保存
		self.beta = graph_reg_beta

		print(f"GraphLearningLoss: L_main = {main_loss_name}, β = {graph_reg_beta}")

	def _get_graph_reg_loss(self, model: nn.Module) -> torch.Tensor:
		"""
		モデルから 'latest_graph_reg_loss' を安全に取得する。
		DataParallel や DDP (DistributedDataParallel) に対応。

		"""
		if hasattr(model, 'module'):
			# DataParallel や DDP の場合
			return model.module.latest_graph_reg_loss
		else:
			# 通常のモデルの場合
			return model.latest_graph_reg_loss

	def forward(self,
	            preds: torch.Tensor,
	            target: torch.Tensor,
	            model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		総損失を計算する。

		Args:
			preds (torch.Tensor): モデルの出力
			target (torch.Tensor): 教師データ
			model (nn.Module):
				学習中のモデルインスタンス (L_regを取得するために必要)

		Returns:
			Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
				- loss_total (torch.Tensor): L_main + β * L_reg
				- loss_main (torch.Tensor): 主損失
				- loss_reg (torch.Tensor): グラフ正則化損失
		"""

		# 1. 主損失 (L_main) の計算
		loss_main = self.main_loss_computer(preds, target)

		# 2. グラフ正則化損失 (L_reg) の取得
		loss_reg = self._get_graph_reg_loss(model)

		# 3. 総損失 (L_total) の計算
		# L_total = L_main + β * L_reg
		#
		loss_total = loss_main + self.beta * loss_reg

		# ログ記録のために個別の損失も返す
		return loss_total, loss_main, loss_reg