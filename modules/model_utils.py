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
  AVOID_REPEAT_TOKENS = []
  CHUNK_LEN = 256
  END_OF_TEXT = 0
  END_OF_LINE = 187
  CHAT_LEN_SHORT = 40
  CHAT_LEN_LONG = 150
  all_state = {}
  
  def __init__(self, args):
    self.model_path = args.model
    self.strategy = args.strategy

  def load_model(self):
    self.model = RWKV(model=self.model_path, strategy=self.strategy)
    self.pipeline = PIPELINE(self.model, f"./20B_tokenizer.json")
    AVOID_REPEAT = '，：？！'
    for i in AVOID_REPEAT:
      dd = self.pipeline.encode(i)
      assert len(dd) == 1
      self.AVOID_REPEAT_TOKENS += dd

  def run_rnn(self, model_tokens, model_state, tokens, newline_adj = 0):
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    while len(tokens) > 0:
      out, model_state = self.model.forward(tokens[:self.CHUNK_LEN], model_state)
      tokens = tokens[self.CHUNK_LEN:]
    out[self.END_OF_LINE] += newline_adj # adjust \n probability
    if model_tokens[-1] in self.AVOID_REPEAT_TOKENS:
      out[model_tokens[-1]] = -999999999
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
  
  def get_reply(self, model_tokens, model_state, out, x_temp, x_top_p, presence_penalty, frequency_penalty):
    new_reply = ''
    begin = len(model_tokens)
    out_last = begin
    occurrence = {}
    for i in range(999):
      if i <= 0:
        newline_adj = -999999999
      elif i <= self.CHAT_LEN_SHORT:
        newline_adj = (i - self.CHAT_LEN_SHORT) / 10
      elif i <= self.CHAT_LEN_LONG:
        newline_adj = 0
      else:
        newline_adj = (i - self.CHAT_LEN_LONG) * 0.25 # MUST END THE GENERATION
      for n in occurrence:
        out[n] -= (presence_penalty + occurrence[n] * frequency_penalty)
      token = self.pipeline.sample_logits(out, temperature=x_temp, top_p=x_top_p)
      if token not in occurrence:
        occurrence[token] = 1
      else:
        occurrence[token] += 1
      
      out, model_tokens, model_state = self.run_rnn(model_tokens, model_state, [token], newline_adj=newline_adj)
      out[self.END_OF_TEXT] = -999999999  # disable <|endoftext|>

      xxx = self.pipeline.decode(model_tokens[out_last:])
      if '\ufffd' not in xxx: # avoid utf-8 display issues
        new_reply += xxx
        out_last = begin + i + 1
    
      send_msg = self.pipeline.decode(model_tokens[begin:])
      if '\n\n' in send_msg:
        send_msg = send_msg.strip()
        break
    if len(model_tokens) > 1000:
      model_tokens = model_tokens[len(model_tokens) - 1000:]
    return new_reply, out, model_tokens, model_state