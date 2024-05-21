import math
import torch
from torch import Tensor
import numpy as np

class Mirostat(object):

  def __init__(self, tau: float=3.0, rate: float=0.1):
    self.set_param(tau, rate, 2 * tau)
  
  def set_param(self, tau: float=3.0, rate: float=0.1, max_surprise = 6.0):
    self.tau = tau
    self.max_surprise = max_surprise
    self.rate = rate

  def choise(self, out: Tensor, min_p) -> int:
    sorted_logits, sorted_indices = torch.sort(out, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1).tolist()
    for i, candidate in enumerate(prob_original):
      if candidate > 0 and -math.log2(candidate) > self.max_surprise:
        if (i == 0):
          sorted_logits = sorted_logits[:1]
        else:
          sorted_logits = sorted_logits[:i]
        break
    prob_topk = torch.softmax(sorted_logits, dim=-1)
    if min_p > 0 and min_p < 1:
      prob_topk[prob_topk < prob_topk[0] * min_p] = 0
    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
    prev = sorted_indices[prev_i]
    observed_surprise = -math.log2(prob_original[prev_i])
    error_surprise = observed_surprise - self.tau
    self.max_surprise -= self.rate * error_surprise
    self.max_surprise = min(self.max_surprise, 4 * self.tau)
    return int(prev[0])