from modules.model_utils import ModelUtils
from pathlib import Path
import os, json, datetime
import pickle
import copy

class Chat:
  
  model_utils = None
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
    model_tokens = []
    model_state = None
    self.user = user
    self.bot = bot
    self.greeting = greeting
    self.bot_persona = bot_persona
    init_prompt = f"请记住你叫{bot}，{bot_persona}你称呼我为{user}，你的描述应当详细冗长,你的语言应当生动且富有创意。\n\n"
    if greeting:
      init_prompt += f"{bot}: 好的，{greeting}"
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
      init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n'.join(init_prompt).strip() + '\n\n'
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat('', 'chat_init', out, model_tokens, model_state)
    if os.path.exists(f'save/{bot}.sav'):
      data = self.__load_chat()
      self.model_utils.save_all_stat(self.srv_chat, 'chat', data['out'], data['model_tokens'], data['model_state'])
      if data['model_tokens_pre']:
        self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', data['out_pre'], data['model_tokens_pre'], data['model_state_pre'])
      self.chatbot = data['chatbot']
    else:
      self.chatbot = [[None, greeting]]
      self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    return self.__generate_cai_chat_html()
  
  def reset_bot(self):
    log_name = f'./log/{self.bot}.json'
    if os.path.exists(log_name):
      os.makedirs(f'./log/{self.bot}', exist_ok=True)
      log_bk_name = f'./log/{self.bot}/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
      os.rename(log_name, log_bk_name)
    out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    try:
      self.model_utils.remove_stat(self.srv_chat, 'chat_pre')
    except:
      pass
    self.chatbot = [[None, self.greeting]]
    save_file = f'save/{self.bot}.sav'
    if os.path.exists(save_file):
      os.remove(save_file)
    return None, self.__generate_cai_chat_html()
  
  def regen_msg(self, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    except:
      return '', self.__generate_cai_chat_html()
    new = f"{self.user}: {self.chatbot[-1][0]}\n\n{self.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    return '', self.gen_msg(out, chat_param, model_tokens, model_state) 
  
  def on_message(self, message, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    if not message:
      return '', self.__generate_cai_chat_html()
    message = message.strip().replace('\r\n','\n').replace('\n\n','\n')
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    if len(model_tokens) > 3000:
      out, model_tokens, model_state = self.arrange_token()
    self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    new = f"{self.user}: {message}\n\n{self.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    self.chatbot += [[message, None]]
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    return '', self.gen_msg(out, chat_param, model_tokens, model_state)
  
  def gen_msg(self, out, chat_param, model_tokens, model_state):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    self.chatbot[-1][1] = new_reply
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    self.__save_log()
    self.__save_chat()
    return self.__generate_cai_chat_html()
    
  def get_prompt(self, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    new = f"{self.user}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    return new_prompt[0]
  
  def clear_last(self):
    if(len(self.chatbot) == 1):
      return self.__generate_cai_chat_html(), ''
    message = self.chatbot[-1][0]
    self.chatbot = self.chatbot[:-1]
    if len(self.chatbot) < 2:
      out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
      self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
      self.model_utils.remove_stat(self.srv_chat, 'chat_pre')
    elif len(self.chatbot) < 3:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
      self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
      out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
      self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    else:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
      self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
      chat_str = self.__get_chatbot_str(self.chatbot[1:-1])
      out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
      self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    self.__save_chat()
    self.__save_log()
    return self.__generate_cai_chat_html(), message
  
  def __save_log(self):
    os.makedirs('log', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in self.chatbot]
    with open(f'./log/{self.bot}.json', 'w', encoding='utf-8') as f:
      json.dump(dict_list, f, ensure_ascii=False, indent=2)

  def __save_chat(self):
    os.makedirs('save', exist_ok=True)
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    try:
      out_pre, model_tokens_pre, model_state_pre = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    except:
      out_pre = None
      model_tokens_pre = None
      model_state_pre = None
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
        <div class="message message_c">
          <div class="circle-bot">
            {img_bot}
          </div>
          <div class="text_c">
            <div class="username">
              {self.bot}
            </div>
            <div class="message-body message-body-c">
              {row[1]}
            </div>
          </div>
        </div>
      """
      if row[0] != None:
        row[0] = row[0].replace('\n', '<br/>')
        output += f"""
          <div class="message message_m">
            <div class="text_m">
              <div class="username username-m">
                {self.user}
              </div>
              <div class="message-body message-body-m">
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
      chat_str += f'{self.user}: {row[0]}\n\n'
      chat_str += f'{self.bot}: {row[1]}\n\n'
    return chat_str
  
  def get_test_data(self):
    data_now = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    txt_now = f"一共：{len(data_now[1])}个token。\n\n{self.model_utils.pipeline.decode(data_now[1])}"
    try:
      data_pre = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
      txt_pre = f"一共：{len(data_pre[1])}个token。\n\n{self.model_utils.pipeline.decode(data_pre[1])}"
    except:
      txt_pre = ''
    return txt_now, txt_pre
  
  def arrange_token(self):
    out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
    chat_str = ''
    chat_str_pre = ''
    i = 0
    for row in reversed(self.chatbot[:-1]):
      if len(chat_str_pre) > 400:
        break
      chat_str_pre = f'{self.bot}: {row[1]}\n\n' + chat_str_pre
      chat_str_pre = f'{self.user}: {row[0]}\n\n' + chat_str_pre
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str_pre))
    self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    chat_str += f'{self.user}: {self.chatbot[-1][0]}\n\n'
    chat_str += f'{self.bot}: {self.chatbot[-1][1]}\n\n'
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    return out, model_tokens, model_state