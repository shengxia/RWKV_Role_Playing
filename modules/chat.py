from modules.model_utils import ModelUtils
import os, gc, json, datetime
import torch
import pickle

class Chat:
  
  model_utils = None
  log_name = ''
  srv_chat = 'chat_server'

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils

  def load_init_prompt(self, user, bot, greeting, bot_persona):
    self.log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    model_tokens = []
    model_state = None
    init_prompt = f"你是{bot}，{bot_persona}，{bot}称呼我为{user}。\n"
    init_prompt += f"{bot}: 我的名字叫{bot}，你叫什么名字？\n"
    init_prompt += f"{user}: 你可以叫我{user}。\n"
    init_prompt += f"{bot}: 你好啊，{user}，很高兴认识你。\n"
    if greeting:
      init_prompt += f"{user}: 我也很高兴认识你，{bot}。\n"
      init_prompt += f"{bot}: {greeting}\n{user}: "
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
      init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n'.join(init_prompt).strip()
    init_prompt = init_prompt.replace('\n\n', '\n')
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat('', 'chat_init', out, model_tokens, model_state)
    chatbot = [[None, greeting]]
    if os.path.exists(f'save/{bot}.sav'):
      data = self.load_chat(bot)
      self.model_utils.save_all_stat(self.srv_chat, 'chat', data['out'], data['model_tokens'], data['model_state'])
      chatbot = data['chatbot']
    else:
      self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    gc.collect()
    torch.cuda.empty_cache()
    return chatbot
  
  def reset_bot(self, greeting, bot):
    self.log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    chatbot = [[None, greeting]]
    self.save_chat(chatbot, bot)
    return None, chatbot
  
  def regen_msg(self, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    except:
      return '', chatbot
    return self.gen_msg(out, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot, model_tokens, model_state)
  
  def on_message(self, message, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot):
    msg = message.replace('\\n','\n').strip()
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    new = f" {msg}\n{bot}: "
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new), newline_adj=-999999999)
    self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    chatbot = chatbot + [[msg, None]]
    return self.gen_msg(out, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot, model_tokens, model_state) 
  
  def gen_msg(self, out, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot, model_tokens, model_state):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, temperature, top_p, presence_penalty, frequency_penalty, user, bot)
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    chatbot[-1][1] = new_reply.replace('\n', '')
    self.save_log(chatbot)
    self.save_chat(chatbot, bot)
    return '', chatbot

  def save_log(self, chatbot):
    os.makedirs('log', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in chatbot]
    with open(f'log/{self.log_name}', 'w', encoding='utf-8') as f:
      json.dump(dict_list, f, ensure_ascii=False, indent=2)

  def get_prompt(self, top_p, temperature, presence_penalty, frequency_penalty, user, bot):
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, temperature, top_p, presence_penalty, frequency_penalty, user, bot)
    return new_prompt.replace('\n', '')
  
  def save_chat(self, chatbot, bot):
    os.makedirs('save', exist_ok=True)
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    data = {
      "out": out,
      "model_tokens": model_tokens,
      "model_state": model_state,
      "chatbot": chatbot
    }
    with open(f'save/{bot}.sav', 'wb') as f:
      pickle.dump(data, f)

  def load_chat(self, bot):
    with open(f'save/{bot}.sav', 'rb') as f:
      data = pickle.load(f)
    return data
  