from modules.model_utils import ModelUtils
from pathlib import Path
import os, json, pickle, copy, re, uuid

class RoleInfo:

  def __init__(self, chatbot, user, bot, action_start, action_end, greeting, use_qa, log_hash):
    self.chatbot = chatbot
    self.user_chat = user
    self.bot_chat = bot
    self.user = user if not use_qa else 'User'
    self.bot = bot if not use_qa else 'Assistant' 
    self.action_start = action_start 
    self.action_start_token = None
    self.action_end = action_end 
    self.action_end_token = None
    self.greeting = greeting 
    self.log_hash = log_hash 
  

class Chat:
  
  model_utils = None
  chat_css = ''
  muti_user = False
  lang = ''
  role_info = {}

  def __init__(self, model_utils:ModelUtils, muti_user, lang):
    self.model_utils = model_utils
    self.muti_user = muti_user
    self.lang = lang
    with open('./css/chat.css', 'r') as f:
      self.chat_css = f.read()

  def load_init_prompt(self, user, bot, action_start, action_end, greeting, bot_persona, example_message, use_qa, as_default=False):
    model_tokens = []
    model_state = None
    self.role_info = RoleInfo([], user, bot, action_start, action_end, greeting, use_qa, str(uuid.uuid1()).replace('-', ''))
    init_prompt = self.__get_init_prompt(bot, bot_persona, user, example_message, as_default)
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
      init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n'.join(init_prompt).strip() + '\n\n'
    if greeting:
      init_prompt += f"{self.role_info.bot}: {greeting}\n\n"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat('chat_init', out, model_tokens, model_state)
    if not self.muti_user and os.path.exists(f'save/{bot}.sav'):
      data = self.__load_chat(self.role_info.bot_chat)
      self.model_utils.save_all_stat('chat', data['out'], data['model_tokens'], data['model_state'])
      if data['model_tokens_pre']:
        self.model_utils.save_all_stat('chat_pre', data['out_pre'], data['model_tokens_pre'], data['model_state_pre'])
      self.role_info.chatbot = data['chatbot']
    else:
      if greeting:
        self.role_info.chatbot = [[None, greeting]]
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    return self.__generate_cai_chat_html()
  
  def reset_bot(self):
    out, model_tokens, model_state = self.model_utils.load_all_stat('chat_init')
    self.role_info.log_hash = str(uuid.uuid1()).replace('-', '')
    self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    try:
      self.model_utils.remove_stat('chat_pre')
    except:
      pass
    if self.role_info.greeting:
      self.role_info.chatbot = [[None, self.role_info.greeting]]
    else:
      self.role_info.chatbot = []
    save_file = f'save/{self.role_info.bot_chat}.sav'
    if os.path.exists(save_file):
      os.remove(save_file)
    return None, None, self.__generate_cai_chat_html()
  
  def regen_msg(self, top_p, temperature, presence_penalty, frequency_penalty, min_len):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
    except:
      return '', self.__generate_cai_chat_html()
    new = f"{self.role_info.user}: {self.role_info.chatbot[-1][0]}\n\n{self.role_info.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(
      top_p, temperature, presence_penalty, frequency_penalty, 
      min_len, self.role_info.action_start_token, self.role_info.action_end_token
    )
    occurrence = self.__get_occurrence(True)
    reply_text = self.gen_msg(out, chat_param, model_tokens, model_state, occurrence) 
    return '', '', reply_text
  
  def on_message(self, message, action, top_p, temperature, presence_penalty, frequency_penalty, action_front, min_len, replace_message):
    message = message.strip().replace('\r\n','\n') if message else ''
    action = action.strip().replace('\r\n','\n') if action else ''
    msg = f"{message}"
    if action_front:
      if action:
        msg = f"{self.role_info.action_start}{action}{self.role_info.action_end}{msg}"
    else:
      if action:
        msg += f"{self.role_info.action_start}{action}{self.role_info.action_end}"
    if replace_message:
      try:
        out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
      except:
        return '', '', self.__generate_cai_chat_html()
      new = f"{self.role_info.user}: {self.role_info.chatbot[-1][0]}\n\n{self.role_info.bot}: {msg}\n\n"
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
      self.role_info.chatbot[-1][1] = msg
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
      return '', '', self.__generate_cai_chat_html()
    else:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat')
      self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
      new = f"{self.role_info.user}: "
      new += f"{msg}\n\n{self.role_info.bot}:"
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
      self.role_info.chatbot += [[msg, None]]
      chat_param = self.model_utils.format_chat_param(
        top_p, temperature, presence_penalty, frequency_penalty, 
        min_len, self.role_info.action_start_token, self.role_info.action_end_token
      )
      occurrence = self.__get_occurrence()
      reply_text = self.gen_msg(out, chat_param, model_tokens, model_state, occurrence)
      return '', '', reply_text
  
  def gen_msg(self, out, chat_param, model_tokens, model_state, occurrence):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param, occurrence)
    self.role_info.chatbot[-1][1] = new_reply
    self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    self.__save_log()
    self.__save_chat()
    return self.__generate_cai_chat_html()
    
  def get_prompt(self, top_p, temperature, presence_penalty, frequency_penalty):
    out, model_tokens, model_state = self.model_utils.load_all_stat('chat')
    new = f"{self.role_info.user}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(top_p, temperature, presence_penalty, frequency_penalty)
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    pos_arr = list(self.__find_all_chat(new_prompt[0]))
    chat_action_data = self.__format_chat_action(pos_arr, new_prompt[0])
    chat, action = self.__get_chat_action(chat_action_data)
    return chat, action
  
  def clear_last(self):
    n = 1
    if(len(self.role_info.chatbot) == 0):
      return self.__generate_cai_chat_html(), '', ''
    if not self.role_info.chatbot[0][0]:
      n += 1
      if(len(self.role_info.chatbot) == 1):
        return self.__generate_cai_chat_html(), '', ''
    messages = self.role_info.chatbot.pop()    
    if len(self.role_info.chatbot) < n:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_init')
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
      self.model_utils.remove_stat('chat_pre')
    elif len(self.role_info.chatbot) < n + 1:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_init')
      self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
    else:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
      chat_str = self.__get_chatbot_str(self.role_info.chatbot[1:-1])
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_init')
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
      self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
    self.__save_chat()
    self.__save_log()
    pos_arr = list(self.__find_all_chat(messages[0]))
    chat_action_data = self.__format_chat_action(pos_arr, messages[0])
    chat, action = self.__get_chat_action(chat_action_data)
    return self.__generate_cai_chat_html(), chat, action
  
  def __save_log(self):
    os.makedirs(f'log/{self.role_info.bot_chat}/', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in self.role_info.chatbot]
    with open(f'./log/{self.role_info.bot_chat}/{self.role_info.log_hash}.json', 'w', encoding='utf-8') as f:
      json.dump(dict_list, f, ensure_ascii=False, indent=2)

  def __save_chat(self):
    if self.muti_user:
      return
    os.makedirs('save', exist_ok=True)
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
    with open(f'save/{self.role_info.bot_chat}.sav', 'wb') as f:
      pickle.dump(data, f)

  def __load_chat(self):
    with open(f'save/{self.role_info.bot_chat}.sav', 'rb') as f:
      data = pickle.load(f)
    return data
  
  def __generate_cai_chat_html(self):
    output = f'<style>{self.chat_css}</style><div class="chat" id="chat">'
    img_bot = f'<img src="file/chars/{self.role_info.bot_chat}.png">' if Path(f'chars/{self.role_info.bot_chat}.png').exists() else ''
    chatbot = copy.deepcopy(self.role_info.chatbot)
    chatbot.reverse()
    for row in chatbot:
      pos_arr = list(self.__find_all_chat(row[1]))
      chat_action_data = self.__format_chat_action(pos_arr, row[1])
      msg = self.__format_chat_html(chat_action_data).replace('\n', '<br>')
      output += f"""
        <div class="message message_c">
          <div class="circle-bot">
            {img_bot}
          </div>
          <div class="text_c">
            <div class="username">
              {self.role_info.bot_chat}
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
        msg = self.__format_chat_html(chat_action_data).replace('\n', '<br>')
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
    output += "</div>"
    return output.replace('<br><br>', '<br>').replace('<br><br>', '<br>')
  
  def __get_chatbot_str(self, chatbot):
    chat_str = ''
    for row in chatbot:
      chat_str += f'{self.role_info.user}: {row[0]}\n\n'
      chat_str += f'{self.role_info.bot}: {row[1]}\n\n'
    return chat_str
  
  def __get_init_prompt(self, bot, bot_persona, user, example_message, as_default=False):
    if not as_default:
      if self.role_info.action_start and self.role_info.action_start in example_message and self.role_info.action_end in example_message:
        self.role_info.action_start_token = self.model_utils.pipeline.encode(f' {self.role_info.action_start}')
        self.role_info.action_end_token = self.model_utils.pipeline.encode(self.role_info.action_end)
      else:
        self.role_info.action_start_token = None
        self.role_info.action_end_token = None
      em = example_message.replace('<bot>', bot).replace('<user>', user)
      init_prompt = {
        'zh': f"阅读并理解以下{user}和{bot}之间的对话：",
        'en': f"The following is a coherent verbose detailed conversation between {user} and {bot}."
      }
      init_prompt_part2 = {
        'zh': f"根据以下描述来扮演{bot}和{user}对话，在对话中加入描述角色的感情、想法、身体动作等内容，也可以加入对环境、场面或动作产生结果的描述，以此来促进对话的进展，这些描述要合理且文采斐然。\n",
        'en': f"The following is another coherent verbose detailed conversation between {user} and {bot}.\n"
      }
      init_prompt_final = init_prompt[self.lang]
      init_prompt_part2_final = init_prompt_part2[self.lang]
      if em:
        init_prompt_final += f'\n\n{em}\n\n{init_prompt_part2_final}'
      else:
        init_prompt_final = f'{init_prompt_part2_final}'
      init_prompt_final += f"{bot_persona}"
    else:
      init_prompt_final = "User: hi\n\nAssistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
    return init_prompt_final

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
    if len(data[1]) < 4000:
      return False
    return True

  def arrange_token(self):
    out, model_tokens, model_state = self.model_utils.load_all_stat('chat_init')
    chat_str = ''
    chat_str_pre = ''
    for row in reversed(self.role_info.chatbot[:-1]):
      if len(chat_str_pre) > 400:
        break
      chat_str_pre = f'{self.role_info.bot}: {row[1]}\n\n' + chat_str_pre
      chat_str_pre = f'{self.role_info.user}: {row[0]}\n\n' + chat_str_pre
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str_pre))
    self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
    chat_str += f'{self.role_info.user}: {self.role_info.chatbot[-1][0]}\n\n'
    chat_str += f'{self.role_info.bot}: {self.role_info.chatbot[-1][1]}\n\n'
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
    self.model_utils.save_all_stat('chat', out, model_tokens, model_state)

  def __find_all_chat(self, input_str):
    if not self.role_info.action_start or not self.role_info.action_end:
      return (0, len(input_str))
    pattern = re.compile("\\" + self.role_info.action_start + ".*?\\" + self.role_info.action_end)
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
  
  def __get_occurrence(self, is_pre=False):
    chatbot = copy.deepcopy(self.role_info.chatbot)
    if len(chatbot) > 3:
      chatbot = chatbot[-3:]
    if is_pre:
      chatbot = chatbot[:-1]
    occurrence = {}
    for i in chatbot:
      if i[1]:
        bot_token = self.model_utils.pipeline.encode(i[1])
        for t in bot_token:
          for o in occurrence:
            occurrence[o] *= self.model_utils.penalty_decay
          occurrence[t] = 1 + (occurrence[t] if t in occurrence else 0)
    return occurrence