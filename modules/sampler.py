import math
import torch
from torch import Tensor

class Sampler(object):

  def __init__(self, tau: float=3.0, rate: float=0.1, lr_decay: float=0.01,):
    self.set_param(tau, rate, lr_decay, 2 * tau)
  
  def set_param(self, tau: float=3.0, rate: float=0.1, lr_decay: float=0.01, max_surprise = 6.0):
    self.tau = tau
    self.max_surprise = max_surprise
    self.rate = rate
    self.lr_decay = lr_decay

  def choise(self, out: Tensor, min_p, temp):
    sorted_logits, sorted_indices = torch.sort(out, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1).tolist()
    # Mirostat v2
    for i, candidate in enumerate(prob_original):
      if candidate > 0 and -math.log2(candidate) > self.max_surprise:
        if (i == 0):
          sorted_logits = sorted_logits[:1]
        else:
          sorted_logits = sorted_logits[:i]
        break
    prob_topk = torch.softmax(sorted_logits, dim=-1)
    # min p
    if min_p > 0 and min_p < 1:
      prob_topk = prob_topk[prob_topk >= prob_topk[0].item() * min_p]
    # temperature
    if self.max_surprise > 3 * self.tau:
      if temp != 1:
        prob_topk = prob_topk ** (1 / temp)
    else:
      if temp > 1:
        prob_topk = prob_topk * temp
      elif temp < 1:
        prob_topk = prob_topk / temp
    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
    prev = sorted_indices[prev_i]
    observed_surprise = -math.log2(prob_original[prev_i])
    error_surprise = observed_surprise - self.tau
    self.max_surprise -= self.rate * error_surprise
    self.rate = self.rate / (1 + self.lr_decay)
    return int(prev[0])
  
  def k_sampler(self, out: Tensor, k, temp):
    sorted_logits, sorted_indices = torch.sort(out, descending=True)
    sorted_logits = sorted_logits[:k]
    prob_topk = torch.softmax(sorted_logits, dim=-1)
    prob_topk = prob_topk ** (1 / temp)
    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
    prev = sorted_indices[prev_i]
    return int(prev[0])