from modules.model_utils import ModelUtils
from modules.role_info import RoleInfo
from pathlib import Path
import os, json, pickle, copy, re, uuid

class Chat:
  
  model_utils = None
  chat_css = ''
  lang = ''
  role_info = None
  chunked_index = None
  chat_length = 4000
  autosave = False

  def __init__(self, model_utils:ModelUtils, lang, chat_length, autosave):
    self.model_utils = model_utils
    self.lang = lang
    self.autosave = autosave
    self.chat_length = chat_length
    with open('./css/chat.css', 'r') as f:
      self.chat_css = f.read()
  
  def load_init_prompt(self, file_name, user, bot, greeting, bot_persona, example_message, use_qa):
    model_tokens = []
    model_state = None
    self.chunked_index = None
    self.role_info = RoleInfo(file_name, [], user, bot, greeting, bot_persona, example_message, 
                              use_qa, str(uuid.uuid1()).replace('-', ''))
    try:
      self.model_utils.remove_stat('chat_pre')
    except:
      pass
    save_file = f'save/{file_name}.sav'
    if os.path.exists(save_file):
      self.load_state(file_name)
    else:
      out, model_tokens, model_state = self.__get_init_state()
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    return self.__generate_cai_chat_html()

  def load_state(self, file_name:str):
    data = self.__load_chat_from(file_name)
    self.model_utils.save_all_stat('chat', data['out'], data['model_tokens'], data['model_state'])
    if data['model_tokens_pre']:
      self.model_utils.save_all_stat('chat_pre', data['out_pre'], data['model_tokens_pre'], data['model_state_pre'])
    self.role_info.chatbot = data['chatbot']
    return self.role_info.chatbot, self.role_info.bot_chat
  
  def reset_bot(self):
    out, model_tokens, model_state = self.__get_init_state()
    self.role_info.log_hash = str(uuid.uuid1()).replace('-', '')
    self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    try:
      self.model_utils.remove_stat('chat_pre')
    except:
      pass
    if self.role_info.greeting:
      self.role_info.chatbot = self.role_info.greeting_chatbot.copy()
    else:
      self.role_info.chatbot = []
    save_file = f'save/{self.role_info.file_name}.sav'
    if os.path.exists(save_file):
      os.remove(save_file)
    self.chunked_index = None
    return None, self.__generate_cai_chat_html(), self.role_info.bot_chat

  def regen_msg(self, speak_to, tau, lr, lr_decay, min_p, temp, presence_penalty):
    if self.chunked_index:
      self.__flush_chat()
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
    except:
      return '', self.__generate_cai_chat_html()
    user_msg = self.role_info.chatbot[-1][0]['msg']
    if self.role_info.use_qa:
      speak_to = self.role_info.bot
    if user_msg:
      new = f'{self.role_info.user}: {user_msg}\n\n{speak_to}:'
    else:
      new = f'{speak_to}:'
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(tau, lr, lr_decay, min_p, temp, presence_penalty)
    reply_text = self.__gen_msg(speak_to, out, chat_param, model_tokens, model_state) 
    return '', reply_text, speak_to
  
  def on_message(self, message, speak_to, tau, lr, lr_decay, min_p, temp, presence_penalty, replace_message):
    if self.chunked_index:
      self.__flush_chat()
    if self.role_info.use_qa:
      speak_to = self.role_info.bot
    msg = message.strip().replace('\r\n','\n') if message else ''
    if replace_message:
      try:
        out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
      except:
        return '', self.__generate_cai_chat_html(), speak_to
      new = f"{self.role_info.user}: {self.role_info.chatbot[-1][0]}\n\n{self.role_info.bot}: {msg}\n\n"
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
      self.role_info.chatbot[-1][1] = {'char': speak_to, 'msg': msg}
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
      self.ban_tokens = []
      return '', self.__generate_cai_chat_html(), speak_to
    else:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat')
      self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
      if msg:
        new = f"{self.role_info.user}: {msg}\n\n{speak_to}:"
      else:
        new = f"{speak_to}:"
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
      user_msg = {'char': self.role_info.user, 'msg': msg}
      self.role_info.chatbot += [[user_msg, None]]
      chat_param = self.model_utils.format_chat_param(tau, lr, lr_decay, min_p, temp, presence_penalty)
      reply_text = self.__gen_msg(speak_to, out, chat_param, model_tokens, model_state)
      self.ban_tokens = []
      return '', reply_text, speak_to
    
  def __gen_msg(self, speak_to, out, chat_param, model_tokens, model_state):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    self.role_info.chatbot[-1][1] = {'char': speak_to, 'msg': new_reply}
    self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    self.__save_log()
    self.__save_chat()
    return self.__generate_cai_chat_html()
    
  def get_prompt(self, tau, lr, lr_decay, min_p, temp, presence_penalty):
    if self.chunked_index:
      self.__flush_chat()
    out, model_tokens, model_state = self.model_utils.load_all_stat('chat')
    new = f"{self.role_info.user}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(tau, lr, lr_decay, min_p, temp, presence_penalty)
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    return new_prompt[0]
  
  def clear_last(self):
    index = len(self.role_info.chatbot) - 1
    if index <= 0:
      return self.__generate_cai_chat_html(), ''
    self.chunked_index = index
    messages = self.role_info.chatbot.pop()
    return self.__generate_cai_chat_html(), messages[0]['msg'], messages[1]['char']
  
  def __flush_chat(self):
    chatbot = copy.deepcopy(self.role_info.chatbot)
    out, model_tokens, model_state = self.__get_init_state()
    if len(chatbot) <= len(self.role_info.greeting_chatbot):
      self.model_utils.remove_stat('chat_pre')
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    else:
      # 全量生成，主要慢在这里
      if chatbot[len(self.role_info.greeting_chatbot):-1]:
        chat_str = self.__get_chatbot_str(chatbot[len(self.role_info.greeting_chatbot):-1])
        out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
      self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
      # 增量生成
      chat_str2 = self.__get_chatbot_str([chatbot[-1]])
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str2))
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    self.chunked_index = None

  def __save_log(self):
    os.makedirs(f'log/{self.role_info.file_name}/', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in self.role_info.chatbot]
    with open(f'./log/{self.role_info.file_name}/{self.role_info.log_hash}.json', 'w', encoding='utf-8') as f:
      json.dump(dict_list, f, ensure_ascii=False, indent=2)

  def __save_init_state(self, file_name, out, model_tokens, model_state):
    save_path = f"./save/init_state/{file_name}.sav"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
      "out": out,
      "model_tokens": model_tokens,
      "model_state": model_state
    }
    with open(save_path, 'wb') as f:
      pickle.dump(data, f)

  def __get_init_state(self):
    out = ''
    model_tokens = []
    model_state = self.model_utils.init_state
    save_file = f"./save/init_state/{self.role_info.file_name}.sav"
    if os.path.exists(save_file):
      with open(save_file, 'rb') as f:
        data = pickle.load(f)
        out = data['out']
        model_tokens = data['model_tokens']
        model_state = data['model_state']
    else:
      init_prompt = self.__get_init_prompt()
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(init_prompt))
      self.__save_init_state(self.role_info.file_name, out, model_tokens, model_state)
    return out, model_tokens, model_state
  
  def save_chat_to(self, file_name:str):
    save_path = f'save/{file_name}.sav'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out, model_tokens, model_state = self.model_utils.load_all_stat('chat')
    try:
      out_pre, model_tokens_pre, model_state_pre = self.model_utils.load_all_stat('chat_pre')
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
      "chatbot": self.role_info.chatbot
    }
    with open(save_path, 'wb') as f:
      pickle.dump(data, f)

  def __save_chat(self):
    if self.autosave:
      return self.save_chat_to(self.role_info.file_name)

  def __load_chat_from(self, file_name:str):
    with open(f'save/{file_name}.sav', 'rb') as f:
      data = pickle.load(f)
    return data
  
  def check_model_state(self):
    try:
      data = self.model_utils.load_all_stat('chat')
    except:
      return True
    if data[2][0].numel() != self.model_utils.n_embd or len(data[2]) / 3 != self.model_utils.n_layer:
      return False
    return True
  
  def __generate_cai_chat_html(self):
    output = f'<style>{self.chat_css}</style><div class="chat" id="chat">'
    chatbot = copy.deepcopy(self.role_info.chatbot)
    chatbot.reverse()
    chat_length = len(chatbot)
    turn = 0
    for row in chatbot:
      if row[1]:
        if self.role_info.use_qa:
          img_bot = f'<img src="file/chars/{self.role_info.file_name}.png">' if Path(f'chars/{self.role_info.file_name}.png').exists() else ''
          char_name = self.role_info.bot_chat
        else:
          img_bot = f'<img src="file/chars/{row[1]["char"]}.png">' if Path(f'chars/{row[1]["char"]}.png').exists() else ''
          char_name = row[1]['char']
        msg = self.__format_chat(row[1]['msg'].replace('\n', '<br>')).replace('<pre><br>', '<pre>')
        output += f"""
          <div class="message message_c">
            <div class="circle-bot">
              {img_bot}
            </div>
            <div class="text_c">
              <div class="username">
                {char_name}<span class="turn">({chat_length - turn}/{chat_length})</span>
              </div>
              <div class="message-body message-body-c">
                {msg}
              </div>
            </div>
          </div>
        """
      if row[0] and row[0]['msg']:
        msg = self.__format_chat(row[0]['msg'].replace('\n', '<br>')).replace('<pre><br>', '<pre>')
        output += f"""
          <div class="message message_m">
            <div class="text_m">
              <div class="username username-m">
                {self.role_info.user_chat}
              </div>
              <div class="message-body message-body-m">
                {msg}
              </div>
            </div>
          </div>
        """
      turn += 1
    output += "</div>"
    return output
  
  def __get_chatbot_str(self, chatbot):
    chat_str = ''
    for row in chatbot:
      if row[0]:
        chat_str += f"{row[0]['char']}: {row[0]['msg']}\n\n"
      if row[1]:
        chat_str += f"{row[1]['char']}: {row[1]['msg']}\n\n"
    return chat_str
  
  def __get_init_prompt(self):
    bot_name = self.role_info.get_pure_char_name()
    em = self.role_info.example_message.replace(
      "{{char}}", bot_name).replace(
      "{{user}}", self.role_info.user_chat)
    bp = self.role_info.bot_persona.replace(
      "{{char}}", bot_name).replace(
      "{{user}}", self.role_info.user_chat)
    greeting = self.__get_chatbot_str(self.role_info.greeting_chatbot)
    init_prompt = ''
    if em:
      init_prompt += f'{em}\n\n'
    init_prompt += f"{bp}"
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
      init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n'.join(init_prompt).strip() + '\n\n'
    if greeting:
      init_prompt += f"{greeting}"
    return f'{init_prompt}'

  def get_test_data(self):
    data_now = self.model_utils.load_all_stat('chat') 
    txt_now = f"token count: {len(data_now[1])}\n\n{self.model_utils.pipeline.decode(data_now[1])}"
    try:
      data_pre = self.model_utils.load_all_stat('chat_pre')
      txt_pre = f"token count: {len(data_pre[1])}\n\n{self.model_utils.pipeline.decode(data_pre[1])}"
    except:
      txt_pre = ''
    return txt_now, txt_pre
  
  def check_token_count(self):
    data = self.model_utils.load_all_stat('chat')
    if len(data[1]) < self.chat_length:
      return False
    return True

  def arrange_token(self):
    out, model_tokens, model_state = self.__get_init_state()
    chat_str = ''
    chat_str_pre = ''
    for row in reversed(self.role_info.chatbot[len(self.role_info.greeting_chatbot):-1]):
      if len(chat_str_pre) > 400:
        break
      chat_str_pre = f"{row[1]['char']}: {row[1]['msg']}\n\n" + chat_str_pre
      chat_str_pre = f"{row[0]['char']}: {row[0]['msg']}\n\n" + chat_str_pre
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str_pre))
    self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
    chat_str += f"{self.role_info.chatbot[-1][0]['char']}: {self.role_info.chatbot[-1][0]['msg']}\n\n"
    chat_str += f"{self.role_info.chatbot[-1][1]['char']}: {self.role_info.chatbot[-1][1]['msg']}\n\n"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
    self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
  
  def __format_chat(self, text):
    pattern1 = re.compile(r'（(.*?)）')
    pattern2 = re.compile(r'\((.*?)\)')
    pattern3 = re.compile(r'\*(.*?)\*')
    pattern4 = re.compile(r'```(.*?)```')
    text1 = re.sub(pattern1, r'<em>\1</em>', text)
    text2 = re.sub(pattern2, r'<em>\1</em>', text1)
    text3 = re.sub(pattern3, r'<em>\1</em>', text2)
    text4 = re.sub(pattern4, r'<pre>\1</pre>', text3)
    return text4
