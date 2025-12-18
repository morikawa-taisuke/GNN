import torch
import torch.nn as nn
from itertools import permutations


def sisnr(x, s, eps=1e-8):
	"""
	calculate training loss
	input:
		  x: separated signal, N x S tensor
		  s: reference signal, N x S tensor
	Return:
		  sisnr: N tensor
	"""

	def l2norm(mat, keepdim=False):
		return torch.norm(mat, dim=-1, keepdim=keepdim)

	if x.shape != s.shape:
		raise RuntimeError(
			"Dimention mismatch when calculate si-snr, {} vs {}".format(
				x.shape, s.shape))
	x_zm = x - torch.mean(x, dim=-1, keepdim=True)
	s_zm = s - torch.mean(s, dim=-1, keepdim=True)
	t = torch.sum(
		x_zm * s_zm, dim=-1,
		keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
	return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def si_snr_loss(ests, egs):
	# spks x n x S
	refs = egs
	num_speeker = len(refs)

	def sisnr_loss(permute):
		# for one permute
		return sum(
			[sisnr(ests[s], refs[t])
			 for s, t in enumerate(permute)]) / len(permute)
		# average the value

	# P x N
	N = egs.size(0)
	#print("N", N)
	sisnr_mat = torch.stack(
		[sisnr_loss(p) for p in permutations(range(num_speeker))])
	max_perutt, _ = torch.max(sisnr_mat, dim=0)
	# si-snr
	return -torch.sum(max_perutt) / N

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
		raise RuntimeError(
			"Dimention mismatch when calculate si-sdr, {} vs {}".format(
				x.shape, s.shape))
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


#--- stftMSE 用のラッパー ---
class StftMseLoss(nn.Module):
	def __init__(self, n_fft=1024, device='cpu'):
		super().__init__()
		self.n_fft = n_fft
		self.mse_loss = nn.MSELoss().to(device)

	def forward(self, ests, egs):
		# ests: [B, C, T], egs: [B, C, T] または [B, T]
		# Speech_separation_main.py を参考にstftMSEを実装

		# （例：単純化のためestsとegsが[B, T]と仮定）
		if ests.dim() > 2: ests = ests.squeeze(1)
		if egs.dim() > 2: egs = egs.squeeze(1)

		ests_stft = torch.stft(ests, n_fft=self.n_fft, return_complex=False)
		egs_stft = torch.stft(egs, n_fft=self.n_fft, return_complex=False)

		return self.mse_loss(ests_stft, egs_stft)


# --- 損失関数を取得するファクトリ関数 ---
def get_loss_function(loss_name: str, device: str = 'cpu'):
	""" 設定ファイルの名前（文字列）に基づいて損失関数（のインスタンスまたは関数）を返す """
	if loss_name == "SISDR":
		# si_sdr_loss は関数なので、そのまま返す（またはラッパーで包む）
		# 入力形式（[B, C, T]）に合わせて調整が必要
		return si_sdr_loss  # ※入力形式の調整が必要

	elif loss_name == "waveMSE":
		return nn.MSELoss().to(device)

	elif loss_name == "stftMSE":
		return StftMseLoss(n_fft=1024, device=device)

	else:
		raise ValueError(f"Loss function {loss_name} not recognized.")