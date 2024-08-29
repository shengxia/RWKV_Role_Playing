import copy
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import gc
from modules.sampler import Sampler

class ModelUtils:

  model = None
  pipline = None
  model_path = None
  state_path = None
  strategy = None
  CHUNK_LEN = 100
  END_OF_TEXT = 0
  NEG_INF = -999999999
  AVOID_REPEAT = '，。：？！,.:!?'
  AVOID_REPEAT_TOKENS = []
  all_state = {}
  init_state = None
  sampler = None

  def __init__(self, args):
    self.model_path = args.model
    self.state_path = args.state
    self.strategy = args.strategy
    self.sampler = Sampler()

  def load_model(self):
    self.model = RWKV(model=self.model_path, strategy=self.strategy)
    self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
    self.n_embd = self.model.w['emb.weight'].shape[1]
    n_layer = 0
    keys = list(self.model.w.keys())
    for x in keys:
      layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
      n_layer = max(n_layer, layer_id+1)
    self.n_layer = n_layer
    if self.state_path:
      self.load_state()
    for i in self.AVOID_REPEAT:
      dd = self.pipeline.encode(i)
      assert len(dd) == 1
      self.AVOID_REPEAT_TOKENS += dd
  
  def load_state(self):
    state_raw = torch.load(f'{self.state_path}.pth')
    init_state = [None] * self.n_layer * 3
    for i in range(self.n_layer):
      dd = self.model.strategy[i]
      dev = dd.device
      atype = dd.atype    
      init_state[i*3+0] = torch.zeros(self.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
      init_state[i*3+1] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
      init_state[i*3+2] = torch.zeros(self.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
    self.init_state = init_state

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
  
  def get_reply(self, model_tokens, model_state, out, chat_param):
    self.clear_cache()
    begin = len(model_tokens)
    out_last = begin
    if chat_param['tau'] > 0:
      max_suprise = self.sampler.max_surprise * 0.5 if self.sampler.max_surprise > 4 * chat_param['tau'] else 2 * chat_param['tau']
      self.sampler.set_param(chat_param['tau'], chat_param['lr'], chat_param['lr_decay'], max_suprise)
    occurrence = {}
    for i in range(300):
      for n in occurrence:
        if out[n] > 0:
          out[n] = out[n] / (1 + chat_param['presence_penalty'])
        else:
          out[n] = out[n] * (1 + chat_param['presence_penalty'])
      now_str = self.pipeline.decode(model_tokens[begin:])
      if i < 50:
        if out[261] > 0:
          out[261] = 0 - out[261]
      temp = chat_param['temp']
      if chat_param['tau'] > 0:
        k = 0
        if i == 0:
          k = 2
        elif now_str.endswith('“') or now_str.endswith('（') or now_str.count('"') / 2 != 0 or now_str.count('*') / 2 != 0:
          k = 10
        if k:
          temp = 1000
        token = self.sampler.choise(out, chat_param['min_p'], temp, k)
      else:
        token = self.pipeline.sample_logits(out, temp, chat_param['min_p'])
      if token not in occurrence:
        occurrence[token] = 1
      out, model_tokens, model_state = self.run_rnn(model_tokens, model_state, [token])
      out[self.END_OF_TEXT] = self.NEG_INF
      xxx = self.pipeline.decode(model_tokens[out_last:])
      if '\ufffd' not in xxx: # avoid utf-8 display issues
        out_last = begin + i + 1
      send_msg = self.pipeline.decode(model_tokens[begin:])
      if '\n\n' in send_msg:
        send_msg = send_msg.strip()
        break
    return send_msg, out, model_tokens, model_state
  
  def format_chat_param(self, tau, lr, lr_decay, min_p, temp, presence_penalty):
    chat_param = {
      'tau': tau,
      'lr': lr,
      'lr_decay': lr_decay,
      'min_p': min_p,
      'temp': temp,
      'presence_penalty': presence_penalty
    }
    return chat_param
  
  def clear_cache(self):
    gc.collect()
    torch.cuda.empty_cache()