import copy, os, gc, json, datetime
from modules.options import cmd_opts
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

model = None
pipline = None

END_OF_TEXT = 0
END_OF_LINE = 187
CHAT_LEN_SHORT = 100
CHAT_LEN_LONG = 200
AVOID_REPEAT_TOKENS = []
CHUNK_LEN = 256

model_tokens = []
model_state = None
all_state = {}
srv = 'dummy_server'
log_name = None
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

def load_init_prompt(user, bot, greeting, bot_persona, scenario, example_dialogue, chatbot):
  gc.collect()
  torch.cuda.empty_cache()
  global log_name, model_tokens, model_state
  log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
  model_tokens = []
  model_state = None
  init_prompt = f"接下来，你要扮演一个名为{bot}的角色与{user}对话，以下是{bot}的性格：\n{bot_persona}\n"
  init_prompt += f"你需要参考以下背景故事来展开对话：\n{scenario}\n"
  example_dialogue_merge = example_dialogue + "{{bot}}： " + greeting + "[sep]"
  init_prompt += f"以下是一段{user}和{bot}的示例对话：\n{example_dialogue_merge}".replace('{{user}}', user).replace('{{bot}}', bot)
  init_prompt = init_prompt.strip().split('\n')
  for c in range(len(init_prompt)):
    init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
  init_prompt = '\n' + ('\n'.join(init_prompt)).strip()
  out = run_rnn(pipeline.encode(init_prompt))
  save_all_stat('', 'chat_init', out)
  save_all_stat(srv, 'chat', out)
  chatbot = [[None, greeting]]
  return user, bot, greeting, bot_persona, scenario, example_dialogue, chatbot

def reset_bot(greeting):
  global log_name
  log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
  out = load_all_stat('', 'chat_init')
  save_all_stat(srv, 'chat', out)
  print("Chat reset.")
  return None, [[None, greeting]]

def load_model():
  global model, pipeline, AVOID_REPEAT_TOKENS
  model = RWKV(model=cmd_opts.model, strategy=cmd_opts.strategy)
  pipeline = PIPELINE(model, f"./20B_tokenizer.json")
  AVOID_REPEAT = '，：？！'
  for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

def save_all_stat(srv, name, last_out):
  n = f'{name}_{srv}'
  all_state[n] = {}
  all_state[n]['out'] = last_out
  all_state[n]['rnn'] = copy.deepcopy(model_state)
  all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
  global model_tokens, model_state
  n = f'{name}_{srv}'
  model_state = copy.deepcopy(all_state[n]['rnn'])
  model_tokens = copy.deepcopy(all_state[n]['token'])
  return all_state[n]['out']

def run_rnn(tokens, newline_adj = 0):
  global model_tokens, model_state
  tokens = [int(x) for x in tokens]
  model_tokens += tokens
  while len(tokens) > 0:
    out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
    tokens = tokens[CHUNK_LEN:]
  out[END_OF_LINE] += newline_adj # adjust \n probability
  if model_tokens[-1] in AVOID_REPEAT_TOKENS:
    out[model_tokens[-1]] = -999999999
  return out

def regen_msg(chatbot, top_p, temperature, presence_penalty, frequency_penalty):
  try:
    out = load_all_stat(srv, 'chat_pre')
  except:
    return
  return gen_msg(out, chatbot, top_p, temperature, presence_penalty, frequency_penalty)

def on_message(message, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot):
  msg = message.replace('\\n','\n').strip()
  out = load_all_stat(srv, 'chat')
  new = f"{user}： {msg}[sep]{bot}："
  out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
  save_all_stat(srv, 'chat_pre', out)
  chatbot = chatbot + [[msg, None]]
  return gen_msg(out, chatbot, top_p, temperature, presence_penalty, frequency_penalty) 

def get_prompt(top_p, temperature, presence_penalty, frequency_penalty, user):
  out = load_all_stat(srv, 'chat')
  new = f"{user}： "
  out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
  new_prompt = get_reply(out, temperature, top_p, presence_penalty, frequency_penalty)
  return new_prompt.replace('[sep]', '')

def gen_msg(out, chatbot, top_p, temperature, presence_penalty, frequency_penalty):
  new_reply = get_reply(out, temperature, top_p, presence_penalty, frequency_penalty)
  save_all_stat(srv, 'chat', out)
  chatbot[-1][1] = new_reply.replace('\n', '').replace('[sep]', '')
  save_log(chatbot)
  return '', chatbot

def get_reply(out, x_temp, x_top_p, presence_penalty, frequency_penalty):
  global model_tokens, model_state
  new_reply = ''
  begin = len(model_tokens)
  out_last = begin
  occurrence = {}
  for i in range(999):
    if i <= 0:
      newline_adj = -999999999
    elif i <= CHAT_LEN_SHORT:
      newline_adj = (i - CHAT_LEN_SHORT) / 10
    elif i <= CHAT_LEN_LONG:
      newline_adj = 0
    else:
      newline_adj = (i - CHAT_LEN_LONG) * 0.25 # MUST END THE GENERATION
    for n in occurrence:
      out[n] -= (presence_penalty + occurrence[n] * frequency_penalty)
    token = pipeline.sample_logits(out, temperature=x_temp, top_p=x_top_p)
    if token not in occurrence:
      occurrence[token] = 1
    else:
      occurrence[token] += 1
    
    out = run_rnn([token], newline_adj=newline_adj)
    out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

    xxx = pipeline.decode(model_tokens[out_last:])
    if '\ufffd' not in xxx: # avoid utf-8 display issues
      new_reply += xxx
      out_last = begin + i + 1
  
    send_msg = pipeline.decode(model_tokens[begin:])
    if '[sep]' in send_msg:
      send_msg = send_msg.strip()
      break
  return new_reply

def save_log(chatbot):
  global log_name
  os.makedirs('log', exist_ok=True)
  dict_list = [{'input': q, 'output': a} for q, a in chatbot]
  with open(f'log/{log_name}', 'w', encoding='utf-8') as f:
    json.dump(dict_list, f, ensure_ascii=False, indent=2)