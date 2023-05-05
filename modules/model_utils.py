import copy
import gc
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

class ModelUtils:

  model = None
  pipline = None
  model_path = None
  strategy = None
  CHAT_LEN_SHORT = 80
  CHAT_LEN_LONG = 160
  CHUNK_LEN = 100
  all_state = {}
  END_OF_LINE = 187
  
  def __init__(self, args):
    self.model_path = args.model
    self.strategy = args.strategy

  def load_model(self):
    self.model = RWKV(model=self.model_path, strategy=self.strategy)
    self.pipeline = PIPELINE(self.model, f"./20B_tokenizer.json")

  def run_rnn(self, model_tokens, model_state, tokens):
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    while len(tokens) > 0:
      out, model_state = self.model.forward(tokens[:self.CHUNK_LEN], model_state)
      tokens = tokens[self.CHUNK_LEN:]
    return out, model_tokens, model_state
  
  def save_all_stat(self, srv, name, last_out, model_tokens, model_state):
    n = f'{name}_{srv}'
    self.all_state[n] = {}
    self.all_state[n]['out'] = last_out
    self.all_state[n]['rnn'] = copy.deepcopy(model_state)
    self.all_state[n]['token'] = copy.deepcopy(model_tokens)

  def load_all_stat(self, srv, name):
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(self.all_state[n]['rnn'])
    model_tokens = copy.deepcopy(self.all_state[n]['token'])
    return self.all_state[n]['out'], model_tokens, model_state
  
  def remove_stat(self, srv, name):
    n = f'{name}_{srv}'
    del self.all_state[n]
  
  def get_reply(self, model_tokens, model_state, out, chat_param):
    model_state_pre = copy.deepcopy(model_state)
    stop_word = ['Below is an instruction', 'User:', 'AI:', 'Instruction:', 'Response:', 'Human:', 'Task:', 'Prompt:', 'Bob:', 'Alice:', 'Question:', 'Answer:']
    begin = len(model_tokens)
    out_last = begin
    occurrence = {}
    for i in range(999):
      if i <= 0:
        nl_bias = -float('inf')
      elif i <= self.CHAT_LEN_SHORT:
        nl_bias = (i - self.CHAT_LEN_SHORT) / 10
      elif i <= self.CHAT_LEN_LONG:
        nl_bias = 0
      else:
        nl_bias = (i - self.CHAT_LEN_LONG) * 0.25
      out[self.END_OF_LINE] += nl_bias
      for n in occurrence:
        out[n] -= (chat_param['presence_penalty'] + occurrence[n] * chat_param['frequency_penalty'])
      token = self.pipeline.sample_logits(out, chat_param['temperature'], chat_param['top_p'], chat_param['top_k'])
      if token not in occurrence:
        occurrence[token] = 1
      else:
        occurrence[token] += 1
      out, model_tokens, model_state = self.run_rnn(model_tokens, model_state, [token])
      xxx = self.pipeline.decode(model_tokens[out_last:])
      if '\ufffd' not in xxx: # avoid utf-8 display issues
        out_last = begin + i + 1
      send_msg = self.pipeline.decode(model_tokens[begin:])
      if '\n\n' in send_msg:
        send_msg = send_msg.strip()
        break
      for s in stop_word:
        if send_msg.endswith(s):
          print(f'error:{send_msg}')
          idx = send_msg.find(s)
          send_msg = f" {send_msg[:idx].strip()}"
          tokens = self.pipeline.encode(send_msg + '\n\n')
          out, model_tokens, model_state = self.run_rnn(model_tokens[:begin], model_state_pre, tokens)
          return send_msg, out, model_tokens, model_state
    return send_msg, out, model_tokens, model_state
  
  def format_chat_param(self, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    chat_param = {
      'top_p': top_p,
      'top_k': top_k,
      'temperature': temperature,
      'presence_penalty': presence_penalty,
      'frequency_penalty': frequency_penalty
    }
    return chat_param
  