from modules.model_utils import ModelUtils
import os, gc, json, datetime
import torch

class Chat:
  
  model_utils = None
  model_tokens = []
  model_state = None
  log_name = ''
  srv_chat = 'chat_server'

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils

  def load_init_prompt(self, user, bot, greeting, bot_persona):
    self.log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    self.model_tokens = []
    self.model_state = None
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
    out, self.model_tokens, self.model_state = self.model_utils.run_rnn(self.model_tokens, self.model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat('', 'chat_init', out, self.model_tokens, self.model_state)
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, self.model_tokens, self.model_state)
    gc.collect()
    torch.cuda.empty_cache()
  
  def reset_bot(self, greeting):
    self.log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat('', 'chat_init')
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, self.model_tokens, self.model_state)
    return None, [[None, greeting]]
  
  def regen_msg(self, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot, max_token):
    try:
      out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    except:
      return '', chatbot
    return self.gen_msg(out, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot, max_token)
  
  def on_message(self, message, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot, max_token):
    msg = message.replace('\\n','\n').strip()
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    new = f" {msg}\n{bot}: "
    out, self.model_tokens, self.model_state = self.model_utils.run_rnn(self.model_tokens, self.model_state, self.model_utils.pipeline.encode(new), newline_adj=-999999999)
    self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, self.model_tokens, self.model_state)
    chatbot = chatbot + [[msg, None]]
    return self.gen_msg(out, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot, max_token) 
  
  def gen_msg(self, out, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot, max_token):
    new_reply, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature, top_p, presence_penalty, frequency_penalty, user, bot, max_token)
    self.model_utils.save_all_stat(self.model_tokens, self.model_state, self.srv_chat, 'chat', out)
    chatbot[-1][1] = new_reply.replace('\n', '')
    self.save_log(chatbot)
    return '', chatbot

  def save_log(self, chatbot):
    os.makedirs('log', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in chatbot]
    with open(f'log/{self.log_name}', 'w', encoding='utf-8') as f:
      json.dump(dict_list, f, ensure_ascii=False, indent=2)

  def get_prompt(self, top_p, temperature, presence_penalty, frequency_penalty, user, bot, max_token):
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    new_prompt, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature, top_p, presence_penalty, frequency_penalty, user, bot, max_token)
    return new_prompt.replace('\n', '')
  