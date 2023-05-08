from modules.model_utils import ModelUtils
from pathlib import Path
import os, json, datetime
import pickle
import copy, re

class Chat:
  
  model_utils = None
  srv_chat = 'chat_server'
  chat_css = ''
  chatbot = []
  user = ''
  bot = ''
  action_start = ''
  action_end = ''
  greeting = ''
  bot_persona = ''
  process_flag = False
  lang = None

  def __init__(self, model_utils:ModelUtils, lang):
    self.model_utils = model_utils
    self.lang = lang
    with open('./css/chat.css', 'r') as f:
      self.chat_css = f.read()

  def load_init_prompt(self, user, bot, action_start, action_end, greeting, bot_persona, example_message):
    model_tokens = []
    model_state = None
    self.model_utils.all_state.clear()
    self.user = user
    self.bot = bot
    self.action_start = action_start
    self.action_end = action_end
    self.greeting = greeting
    self.bot_persona = bot_persona
    init_prompt = self.__get_init_prompt(bot, bot_persona, user, example_message)
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
      init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n'.join(init_prompt).strip()
    if greeting:
      init_prompt += f"\n\n{bot}: {greeting}\n\n"
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
    return None, None, self.__generate_cai_chat_html()
  
  def regen_msg(self, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    except:
      return '', self.__generate_cai_chat_html()
    new = f"{self.user}: {self.chatbot[-1][0]}\n\n{self.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    return '', '', self.gen_msg(out, chat_param, model_tokens, model_state) 
  
  def on_message(self, message, action, top_p, top_k, temperature, presence_penalty, frequency_penalty, action_front):
    message = message.strip().replace('\r\n','\n').replace('\n\n','\n') if message else ''
    action = action.strip().replace('\r\n','\n').replace('\n\n','\n') if action else ''
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    new = f"{self.user}: "
    msg = f"{message}"
    if action_front:
      if action:
        msg = f"{self.action_start}{action}{self.action_end}{msg}"
    else:
      if action:
        msg += f"{self.action_start}{action}{self.action_end}"
    new += f"{msg}\n\n{self.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    self.chatbot += [[msg, None]]
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    return '', '', self.gen_msg(out, chat_param, model_tokens, model_state)
  
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
    pos_arr = list(self.__find_all_chat(new_prompt[0]))
    chat_action_data = self.__format_chat_action(pos_arr, new_prompt[0])
    chat, action = self.__get_chat_action(chat_action_data)
    return chat, action
  
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
    pos_arr = list(self.__find_all_chat(message))
    chat_action_data = self.__format_chat_action(pos_arr, message)
    chat, action = self.__get_chat_action(chat_action_data)
    return self.__generate_cai_chat_html(), chat, action
  
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
    chatbot = copy.deepcopy(self.chatbot)
    chatbot.reverse()
    for row in chatbot:
      pos_arr = list(self.__find_all_chat(row[1]))
      chat_action_data = self.__format_chat_action(pos_arr, row[1])
      msg = self.__format_chat_html(chat_action_data)
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
              {msg}
            </div>
          </div>
        </div>
      """
      if row[0] != None:
        pos_arr = list(self.__find_all_chat(row[0]))
        chat_action_data = self.__format_chat_action(pos_arr, row[0])
        msg = self.__format_chat_html(chat_action_data)
        output += f"""
          <div class="message message_m">
            <div class="text_m">
              <div class="username username-m">
                {self.user}
              </div>
              <div class="message-body message-body-m">
                {msg}
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
  
  def __get_init_prompt(self, bot, bot_persona, user, example_message):
    em = example_message.replace('<bot>', bot).replace('<user>', user)
    if self.lang == 'en':
      init_prompt = f"You are {bot}, {bot_persona}\n\n{em}"
    else:
      init_prompt = f"你是{bot}，{bot_persona}\n\n{em}"
    return init_prompt

  def get_test_data(self):
    data_now = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    txt_now = f"token count: {len(data_now[1])}\n\n{self.model_utils.pipeline.decode(data_now[1])}"
    try:
      data_pre = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
      txt_pre = f"token count: {len(data_pre[1])}\n\n{self.model_utils.pipeline.decode(data_pre[1])}"
    except:
      txt_pre = ''
    return txt_now, txt_pre
  
  def check_token_count(self):
    data = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    if len(data[1]) < 5500:
      return False
    return True

  def arrange_token(self):
    self.process_flag = True
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
    self.process_flag = False

  def __find_all_chat(self, input_str):
    if not self.action_start or not self.action_end:
      return (0, len(input_str))
    pattern = re.compile("\\" + self.action_start + ".*?\\" + self.action_end)
    while True:
      match = re.search(pattern, input_str)
      if not match:
        break
      yield match.span()
      input_str = input_str[match.end():]

  def __format_chat_action(self, pos_arr, input_str):
    output_data = []
    for l in pos_arr:
      if l[0] != 0:
        str1 = input_str[:l[0]]
        output_data.append([str1, 'chat'])
      output_data.append([input_str[l[0] + 1:l[1] - 1], 'action'])
      input_str = input_str[l[1]:]
    if str:
      output_data.append([input_str, 'chat'])
    return output_data
    
  def __format_chat_html(self, chat_action_arr):
    output_str = ''
    for ca in chat_action_arr:
      if ca[1] == 'action':
        output_str += f'<i>{ca[0]}</i><br>'
      else:
        output_str += f'{ca[0]}<br>'
    return output_str[:-4]
  
  def __get_chat_action(self, chat_action_arr):
    chat = ''
    action = ''
    for i in chat_action_arr:
      if chat == '':
        if i[1] == 'chat':
          chat = i[0]
      if action == '':
        if i[1] == 'action':
          action = i[0]
    return chat, action