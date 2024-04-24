
import math
import torch
from torch import Tensor
import numpy as np

class Mirostat(object):

  def __init__(self, tau: float=3.0, rate: float=0.1, top_p: float=0.6):
    self.tau = tau
    self.max_surprise = 2 * self.tau
    self.rate = rate
    self.top_p = top_p

  def choise(self, out: Tensor) -> int:
    sorted_logits, sorted_indices = torch.sort(out, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1)
    if self.top_p >= 1 or self.top_p <= 0:
      prob_original = prob_original.tolist()
    else:
      probs = prob_original.cpu().numpy()
      cumulative_probs = np.cumsum(probs)
      cutoff = float(prob_original[np.argmax(cumulative_probs >= self.top_p)])
      prob_original = prob_original[prob_original < cutoff].tolist()
    for i, candidate in enumerate(prob_original):
      if candidate > 0 and -math.log2(candidate) > self.max_surprise:
        if (i == 0):
          sorted_logits = sorted_logits[:1]
        else:
          sorted_logits = sorted_logits[:i]
        break
    prob_topk = torch.softmax(sorted_logits, dim = 0)
    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
    prev = sorted_indices[prev_i]
    topk = prob_topk.cpu().numpy()
    observed_surprise = np.mean(-np.log2(topk))
    # observed_surprise = -math.log2(prob_topk[prev_i])
    error_surprise = observed_surprise - self.tau
    self.max_surprise -= self.rate * error_surprise
    self.max_surprise = min(self.max_surprise, 4 * self.tau)
    return int(prev[0])