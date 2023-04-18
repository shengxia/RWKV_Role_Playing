from modules.model_utils import ModelUtils
from pathlib import Path
import os, gc, json, datetime
import torch
import pickle
import copy

class Chat:
  
  model_utils = None
  log_name = ''
  srv_chat = 'chat_server'
  chat_css = ''
  chatbot = []
  user = ''
  bot = ''
  greeting = ''
  bot_persona = ''

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils
    with open('./css/chat.css', 'r') as f:
      self.chat_css = f.read()

  def load_init_prompt(self, user, bot, greeting, bot_persona):
    self.log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    model_tokens = []
    model_state = None
    self.user = user
    self.bot = bot
    self.greeting = greeting
    self.bot_persona = bot_persona
    init_prompt = f"{user}:你是{bot}，{bot_persona}，{bot}称呼我为{user}。\n"
    if greeting:
      init_prompt += f"{bot}:{greeting}\n{user}:"
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
      init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n'.join(init_prompt).strip()
    init_prompt = init_prompt.replace('\n\n', '\n')
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat('', 'chat_init', out, model_tokens, model_state)
    if os.path.exists(f'save/{bot}.sav'):
      data = self.__load_chat()
      self.model_utils.save_all_stat(self.srv_chat, 'chat', data['out'], data['model_tokens'], data['model_state'])
      self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', data['out_pre'], data['model_tokens_pre'], data['model_state_pre'])
      self.chatbot = data['chatbot']
    else:
      self.chatbot = [[None, greeting]]
      self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    gc.collect()
    torch.cuda.empty_cache()
    return self.__generate_cai_chat_html()
  
  def reset_bot(self):
    self.log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    self.chatbot = [[None, self.greeting]]
    save_file = f'save/{self.bot}.sav'
    if os.path.exists(save_file):
      os.remove(save_file)
    return None, self.__generate_cai_chat_html()
  
  def regen_msg(self, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    except:
      return '', self.chatbot
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    return self.gen_msg(out, chat_param, model_tokens, model_state)
  
  def on_message(self, message, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    new = f"{message}\n{self.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    self.chatbot += [[message, None]]
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    return self.gen_msg(out, chat_param, model_tokens, model_state) 
  
  def gen_msg(self, out, chat_param, model_tokens, model_state):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param, 'chat', self.user, self.bot, 'bot')
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    self.chatbot[-1][1] = new_reply
    self.__save_log()
    self.__save_chat()
    return '', self.__generate_cai_chat_html()

  def get_prompt(self, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, chat_param, 'chat', self.user, self.bot, 'user')
    return new_prompt[0]
  
  def clear_last(self):
    if(len(self.chatbot) < 2):
      return self.__generate_cai_chat_html(), ''
    message = self.chatbot[-1][0]
    self.chatbot = self.chatbot[0:-1]
    # 1. 将前一次的状态赋予本次
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    # 2. 构建前一次的状态，如果对话比较长，这个过程会很慢，非常不建议经常使用该功能
    if len(self.chatbot) < 2:
      out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
      self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    else:
      chat_str = self.__get_chatbot_str(self.chatbot[1])
      out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
      self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    self.__save_chat()
    self.__save_log()
    return self.__generate_cai_chat_html(), message
  
  def __save_log(self):
    os.makedirs('log', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in self.chatbot]
    with open(f'log/{self.log_name}', 'w', encoding='utf-8') as f:
      json.dump(dict_list, f, ensure_ascii=False, indent=2)

  def __save_chat(self):
    os.makedirs('save', exist_ok=True)
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    out_pre, model_tokens_pre, model_state_pre = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    data = {
      "out": out,
      "model_tokens": model_tokens,
      "model_state": model_state,
      "out_pre": out_pre,
      "model_tokens_pre": model_tokens_pre,
      "model_state_pre": model_state_pre,
      "chatbot": self.chatbot
    }
    with open(f'save/{self.bot}.sav', 'wb') as f:
      pickle.dump(data, f)

  def __load_chat(self):
    with open(f'save/{self.bot}.sav', 'rb') as f:
      data = pickle.load(f)
    return data
  
  def __generate_cai_chat_html(self):
    output = f'<style>{self.chat_css}</style><div class="chat" id="chat">'

    img_bot = f'<img src="file/chars/{self.bot}.png">' if Path(f"chars/{self.bot}.png").exists() else ''
    img_me = f'<img src="file/chars/me.png">' if Path(f"chars/me.png").exists() else ''

    chatbot = copy.deepcopy(self.chatbot)
    chatbot.reverse()
    for row in chatbot:
      row[1] = row[1].replace('\n', '<br/>')
      output += f"""
        <div class="message_c">
          <div class="circle-bot">
            {img_bot}
          </div>
          <div class="text">
            <div class="username">
              {self.bot}
            </div>
            <div class="message-body">
              {row[1]}
            </div>
          </div>
        </div>
      """
      if row[0] != None:
        row[0] = row[0].replace('\n', '<br/>')
        output += f"""
          <div class="message_c">
            <div class="circle-you">
              {img_me}
            </div>
            <div class="text">
              <div class="username">
                {self.user}
              </div>
              <div class="message-body">
                {row[0]}
              </div>
            </div>
          </div>
        """
    output += "</div>"
    return output
  
  def __get_chatbot_str(self, chatbot):
    chat_str = ''
    for row in chatbot:
      chat_str += f'{self.user}:{row[0]}\n'
      chat_str += f'{self.bot}:{row[1]}\n'
    return chat_str