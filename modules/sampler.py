import math
import torch
from torch import Tensor

class Sampler(object):

  def __init__(self, tau: float=3.0, rate: float=0.1):
    self.set_param(tau, rate, 2 * tau)
  
  def set_param(self, tau: float=3.0, rate: float=0.1, max_surprise = 6.0):
    self.tau = tau
    self.max_surprise = max_surprise
    self.rate = rate

  def choise(self, out: Tensor, min_p, min_temp, max_temp, dynatemp_exponent):
    if self.tau == 0:
      return (int(sorted_indices[0]), 0, 0)
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
      prob_topk[prob_topk < prob_topk[0].item() * min_p] = 0
    # 动态temperature
    dyn_temp = torch.tensor(0)
    if dynatemp_exponent > 0:
      entropy = -1.0 * torch.where(prob_topk > 0, prob_topk * torch.log2(prob_topk), torch.zeros_like(prob_topk)).sum()
      entropy = max(entropy, torch.tensor(1e-10))
      num_valid_tokens = torch.sum(sorted_logits > -float('inf')).item()
      max_entropy = math.log2(num_valid_tokens)
      max_entropy = max_entropy if max_entropy > 0.0 else 1e-10
      normalized_entropy = entropy / max_entropy  
      dyn_temp = min_temp + (max_temp - min_temp) * (normalized_entropy.pow(dynatemp_exponent))
      prob_topk = prob_topk ** (1 / dyn_temp)
    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
    prev = sorted_indices[prev_i]
    observed_surprise = -math.log2(prob_original[prev_i])
    error_surprise = observed_surprise - self.tau
    self.max_surprise -= self.rate * error_surprise
    return (int(prev[0]), self.max_surprise, dyn_temp.item())