import torch
from torch import Tensor

class Sampler(object):

  def choise(self, out: Tensor, min_p, temp):
    sorted_logits, sorted_indices = torch.sort(out, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1)
    # min p
    if min_p > 0 and min_p < 1:
      prob_original = prob_original[prob_original >= prob_original[0].item() * min_p]
    # temperature
    prob_original = prob_original / temp
    prev_i = torch.multinomial(prob_original, num_samples=1, replacement=True)
    prev = sorted_indices[prev_i]
    return int(prev[0])