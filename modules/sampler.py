import math
import torch
from torch import Tensor
import numpy as np

class Sampler(object):

  def __init__(self, tau: float=3.0, rate: float=0.1, lr_decay: float=0.01,):
    self.set_param(tau, rate, lr_decay, 2 * tau)
  
  def set_param(self, tau: float=3.0, rate: float=0.1, lr_decay: float=0.01, max_surprise = 6.0):
    self.tau = tau
    self.max_surprise = max_surprise
    self.rate = rate
    self.lr_decay = lr_decay

  def choise(self, out: Tensor, top_p, temp, k = 0):
    sorted_logits, sorted_indices = torch.sort(out, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1).tolist()
    if k:
      sorted_logits = sorted_logits[:k]
    else:
      # Mirostat v2
      for i, candidate in enumerate(prob_original):
        if candidate > 0 and -math.log2(candidate) > self.max_surprise:
          if (i == 0):
            sorted_logits = sorted_logits[:1]
          else:
            sorted_logits = sorted_logits[:i]
          break
    prob_topk = torch.softmax(sorted_logits, dim=-1)
    # top p
    if top_p > 0 and top_p < 1 and k == 0:
      cumulative_probs = torch.cumsum(prob_topk, dim=-1).cpu().numpy()
      cutoff = float(prob_topk[np.argmax(cumulative_probs >= top_p)])
      prob_topk = prob_topk[prob_topk >= cutoff]
    # temperature
    if temp != 1:
      prob_topk = prob_topk ** (1 / temp)
    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
    prev = sorted_indices[prev_i]
    observed_surprise = -math.log2(prob_original[prev_i])
    error_surprise = observed_surprise - self.tau
    self.max_surprise -= self.rate * error_surprise
    self.rate = self.rate / (1 + self.lr_decay)
    return int(prev[0])