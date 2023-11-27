import copy
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import gc
import numpy as np
from torch.nn import functional as F

class ModelUtils:

  model = None
  pipline = None
  model_path = None
  strategy = None
  CHUNK_LEN = 100
  END_OF_TEXT = 0
  END_OF_LINE = 11
  DOUBLE_END_OF_LINE = 261
  CHN_PERIOD_END = 28329
  NEG_INF = -999999999
  AVOID_REPEAT = '.!?,()[]{}。！？，（）:：'
  AVOID_REPEAT_TOKENS = []
  all_state = {}

  def __init__(self, args):
    self.model_path = args.model
    self.strategy = args.strategy

  def load_model(self):
    self.model = RWKV(model=self.model_path, strategy=self.strategy)
    self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
    for i in self.AVOID_REPEAT:
      dd = self.pipeline.encode(i)
      assert len(dd) == 1
      self.AVOID_REPEAT_TOKENS += dd

  def run_rnn(self, model_tokens, model_state, tokens):
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    while len(tokens) > 0:
      out, model_state = self.model.forward(tokens[:self.CHUNK_LEN], model_state)
      tokens = tokens[self.CHUNK_LEN:]
    if model_tokens[-1] in self.AVOID_REPEAT_TOKENS:
      out[model_tokens[-1]] = self.NEG_INF
    return out, model_tokens, model_state
  
  def save_all_stat(self, name, last_out, model_tokens, model_state):
    n = f'{name}'
    self.all_state[n] = {
      'out': last_out,
      'rnn': copy.deepcopy(model_state),
      'token': copy.deepcopy(model_tokens)
    }

  def load_all_stat(self, name):
    n = f'{name}'
    model_state = copy.deepcopy(self.all_state[n]['rnn'])
    model_tokens = copy.deepcopy(self.all_state[n]['token'])
    return self.all_state[n]['out'], model_tokens, model_state
  
  def remove_stat(self, name):
    n = f'{name}'
    if n in self.all_state.keys():
      del self.all_state[n]
  
  def get_reply(self, model_tokens, model_state, out, chat_param, occurrence_tokens=[]):
    self.clear_cache()
    begin = len(model_tokens)
    out_last = begin
    if chat_param['force_action']:
      out[23244] = 10
    occurrence = {}
    for t in occurrence_tokens:
      if t in self.AVOID_REPEAT_TOKENS:
        continue
      if t in occurrence:
        occurrence[t] += 1
      else:
        occurrence[t] = 0
    for i in range(500):
      for n in occurrence:
        out[n] -= (chat_param['presence_penalty'] + occurrence[n] * chat_param['frequency_penalty'])
      for xxx in occurrence:
        occurrence[xxx] *= 0.996
      token = self.pipeline.sample_logits(out, chat_param['temperature'], chat_param['top_p'], chat_param['top_k'])
      out, model_tokens, model_state = self.run_rnn(model_tokens, model_state, [token])
      out[self.END_OF_TEXT] = self.NEG_INF
      if token not in self.AVOID_REPEAT_TOKENS:
        if token not in occurrence:
          occurrence[token] = 1
        else:
          occurrence[token] += 1
      xxx = self.pipeline.decode(model_tokens[out_last:])
      if '\ufffd' not in xxx: # avoid utf-8 display issues
        out_last = begin + i + 1
      send_msg = self.pipeline.decode(model_tokens[begin:])
      if '\n\n' in send_msg:
        send_msg = send_msg.strip()
        break
      out = out * 0.996
    return send_msg, out, model_tokens, model_state
  
  def format_chat_param(self, top_p, top_k, temperature, presence_penalty, frequency_penalty, force_action=False):
    chat_param = {
      'top_p': top_p,
      'top_k': top_k,
      'temperature': temperature,
      'presence_penalty': presence_penalty,
      'frequency_penalty': frequency_penalty,
      'force_action': force_action
    }
    return chat_param
  
  def clear_cache(self):
    gc.collect()
    torch.cuda.empty_cache()

  def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    # 'privateuseone' is the type of custom devices like `torch_directml.device()`
    if probs.device.type in ['cpu', 'privateuseone']:
      probs = probs.cpu().numpy()
      sorted_ids = np.argsort(probs)
      sorted_probs = probs[sorted_ids][::-1]
      cumulative_probs = np.cumsum(sorted_probs)
      cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
      probs[probs < cutoff] = 0
      if top_k < len(probs) and top_k > 0:
        probs[sorted_ids[:-top_k]] = 0
      if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
      probs = probs / np.sum(probs)
      out = np.random.choice(a=len(probs), p=probs)
      return int(out)
    else:
      sorted_ids = torch.argsort(probs)
      sorted_probs = probs[sorted_ids]
      sorted_probs = torch.flip(sorted_probs, dims=(0,))
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
      cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
      probs[probs < cutoff] = 0
      if top_k < len(probs) and top_k > 0:
        probs[sorted_ids[:-top_k]] = 0
      if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
      # 修改了一下这里，看看有没有什么变化
      out = torch.multinomial(probs, num_samples=1, replacement=True)[0]
      return int(out)
  