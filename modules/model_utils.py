import copy
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
  CHAT_LEN_LONG = 300
  all_state = {}
  user = "Bob"
  bot = "Alice"
  
  def __init__(self, args):
    self.model_path = args.model
    self.strategy = args.strategy

  def load_model(self):
    self.model = RWKV(model=self.model_path, strategy=self.strategy)
    self.pipeline = PIPELINE(self.model, f"./20B_tokenizer.json")

  def run_rnn(self, model_tokens, model_state, tokens):
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    out, model_state = self.model.forward(tokens, model_state)
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
  
  def get_reply(self, model_tokens, model_state, out, x_temp, x_top_p, x_top_k, presence_penalty, frequency_penalty, user='', bot='', reply_owner='bot'):
    if not user:
      user = self.user
    if not bot:
      bot = self.bot
    new_reply = ''
    begin = len(model_tokens)
    out_last = begin
    occurrence = {}
    for i in range(self.CHAT_LEN_LONG):
      for n in occurrence:
        out[n] -= (presence_penalty + occurrence[n] * frequency_penalty)
      token = self.pipeline.sample_logits(out, temperature=x_temp, top_p=x_top_p, top_k=x_top_k)
      if token not in occurrence:
        occurrence[token] = 1
      else:
        occurrence[token] += 1
      out, model_tokens, model_state = self.run_rnn(model_tokens, model_state, [token])
      xxx = self.pipeline.decode(model_tokens[out_last:])
      if '\ufffd' not in xxx: # avoid utf-8 display issues
        new_reply += xxx
        out_last = begin + i + 1
      send_msg = self.pipeline.decode(model_tokens[begin:])
      if reply_owner == 'bot':
        if send_msg.endswith(f'{user}:'):
          send_msg = send_msg[:-len(f'{user}:')].strip()
          break
      if reply_owner == 'user':
        if send_msg.endswith(f'{bot}:'):
          send_msg = send_msg[:-len(f'{bot}:')].strip()
          break
    return send_msg, out, model_tokens, model_state
  