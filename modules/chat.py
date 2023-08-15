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

  def __init__(self, model_utils:ModelUtils, lang):
    self.model_utils = model_utils
    self.lang = lang
    with open('./css/chat.css', 'r') as f:
      self.chat_css = f.read()
  
  def load_init_prompt(self, file_name, user, bot, action_start, action_end, greeting, bot_persona, example_message, use_qa):
    model_tokens = []
    model_state = None
    self.chunked_index = None
    self.role_info = RoleInfo(file_name, [], user, bot, action_start, action_end, greeting, bot_persona, 
                              example_message, use_qa, str(uuid.uuid1()).replace('-', ''))
    if action_start and action_start in example_message and action_end in example_message:
      self.role_info.action_start_token = self.model_utils.pipeline.encode(f' {action_start}')[0]
      self.role_info.action_end_token = self.model_utils.pipeline.encode(action_end)[0]
    if os.path.exists(f'save/{file_name}.sav'):
      self.load_state(file_name)
    else:
      out, model_tokens, model_state = self.__get_init_state()
      if greeting:
        self.role_info.chatbot = [[None, greeting]]
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    return self.__generate_cai_chat_html()

  def load_state(self, file_name:str):
    data = self.__load_chat_from(file_name)
    self.model_utils.save_all_stat('chat', data['out'], data['model_tokens'], data['model_state'])
    if data['model_tokens_pre']:
      self.model_utils.save_all_stat('chat_pre', data['out_pre'], data['model_tokens_pre'], data['model_state_pre'])
    self.role_info.chatbot = data['chatbot']
    return self.__generate_cai_chat_html()
  
  def reset_bot(self):
    out, model_tokens, model_state = self.__get_init_state()
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
    save_file = f'save/{self.role_info.file_name}.sav'
    if os.path.exists(save_file):
      os.remove(save_file)
    self.chunked_index = None
    return None, None, self.__generate_cai_chat_html()
  
  def regen_msg(self, top_p, tau, temperature, presence_penalty, frequency_penalty, cfg, min_len):
    if self.chunked_index:
      self.__flush_chat()
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
    except:
      return '', '', self.__generate_cai_chat_html()
    new = f"{self.role_info.user}: {self.role_info.chatbot[-1][0]}\n\n{self.role_info.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    out_cfg, token_cfg, state_cfg = self.__get_cfg_state(new, cfg)
    chat_param = self.model_utils.format_chat_param(
      top_p, tau, temperature, presence_penalty, frequency_penalty, cfg,
      min_len, self.role_info.action_start_token, self.role_info.action_end_token
    )
    occurrence = self.__get_occurrence(True)
    reply_text = self.__gen_msg(out, chat_param, model_tokens, model_state, occurrence, out_cfg, token_cfg, state_cfg) 
    return '', '', reply_text
  
  def on_message(self, message, action, top_p, tau, temperature, presence_penalty, frequency_penalty, cfg, action_front, min_len, replace_message):
    if self.chunked_index:
      self.__flush_chat()
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
      out_cfg, token_cfg, state_cfg = self.__get_cfg_state(new, cfg)
      self.role_info.chatbot += [[msg, None]]
      chat_param = self.model_utils.format_chat_param(
        top_p, tau, temperature, presence_penalty, frequency_penalty, cfg,
        min_len, self.role_info.action_start_token, self.role_info.action_end_token
      )
      occurrence = self.__get_occurrence()
      reply_text = self.__gen_msg(out, chat_param, model_tokens, model_state, occurrence, out_cfg, token_cfg, state_cfg)
      return '', '', reply_text
    
  def __get_cfg_state(self, new, cfg):
    out_cfg = None
    token_cfg = None
    state_cfg = None
    if cfg > 0:
      out_cfg, token_cfg, state_cfg = self.__get_init_state()
      out_cfg, token_cfg, state_cfg = self.model_utils.run_rnn(token_cfg, state_cfg, self.model_utils.pipeline.encode(new))
    return out_cfg, token_cfg, state_cfg
  
  def __gen_msg(self, out, chat_param, model_tokens, model_state, occurrence, out_cfg, token_cfg, state_cfg):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param, occurrence, out_cfg, token_cfg, state_cfg)
    self.role_info.chatbot[-1][1] = new_reply
    self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    self.__save_log()
    self.__save_chat()
    return self.__generate_cai_chat_html()
    
  def get_prompt(self, top_p, tau, temperature, presence_penalty, frequency_penalty):
    if self.chunked_index:
      self.__flush_chat()
    out, model_tokens, model_state = self.model_utils.load_all_stat('chat')
    new = f"{self.role_info.user}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(top_p, tau, temperature, presence_penalty, frequency_penalty, 0)
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    pos_arr = list(self.__find_all_chat(new_prompt[0]))
    chat_action_data = self.__format_chat_action(pos_arr, new_prompt[0])
    chat, action = self.__get_chat_action(chat_action_data)
    return chat, action
  
  def clear_last(self):
    index = len(self.role_info.chatbot) - 1
    if index <= 0:
      return self.__generate_cai_chat_html(), '', ''
    self.chunked_index = index
    messages = self.role_info.chatbot.pop()
    pos_arr = list(self.__find_all_chat(messages[0]))
    chat_action_data = self.__format_chat_action(pos_arr, messages[0])
    chat, action = self.__get_chat_action(chat_action_data)
    return self.__generate_cai_chat_html(), chat, action
  
  def __flush_chat(self):
    chatbot = copy.deepcopy(self.role_info.chatbot)
    out, model_tokens, model_state = self.__get_init_state()
    if len(chatbot) < 2:
      self.model_utils.remove_stat('chat_pre')
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    else:
      if len(chatbot) == 2:
        self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
      else:
        # 全量生成，主要慢在这里
        chat_str = self.__get_chatbot_str(chatbot[1:-1])
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
    save_path = f"./chars/init_state/{file_name}.sav"
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
    model_state = None
    save_file = f"./chars/init_state/{self.role_info.file_name}.sav"
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
    return self.save_chat_to(self.role_info.file_name)

  def __load_chat_from(self, file_name:str):
    with open(f'save/{file_name}.sav', 'rb') as f:
      data = pickle.load(f)
    return data
  
  def __generate_cai_chat_html(self):
    output = f'<style>{self.chat_css}</style><div class="chat" id="chat">'
    img_bot = f'<img src="file/chars/{self.role_info.file_name}.png">' if Path(f'chars/{self.role_info.file_name}.png').exists() else ''
    chatbot = copy.deepcopy(self.role_info.chatbot)
    chatbot.reverse()
    for row in chatbot:
      msg = row[1].replace('\n', '').replace(self.role_info.action_start, '<p>').replace(self.role_info.action_end, '</p>')
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
        msg = row[0].replace('\n', '').replace(self.role_info.action_start, '<p>').replace(self.role_info.action_end, '</p>')
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
    return output
  
  def __get_chatbot_str(self, chatbot):
    chat_str = ''
    for row in chatbot:
      chat_str += f'{self.role_info.user}: {row[0]}\n\n'
      chat_str += f'{self.role_info.bot}: {row[1]}\n\n'
    return chat_str
  
  def __get_init_prompt(self):
    em = self.role_info.example_message.replace('<bot>', self.role_info.bot_chat).replace('<user>', self.role_info.user_chat)
    init_prompt = {
      'zh': f"阅读并理解以下{self.role_info.user_chat}和{self.role_info.bot_chat}之间的对话：",
      'en': f"The following is a coherent verbose detailed conversation between {self.role_info.user_chat} and {self.role_info.bot_chat}."
    }
    init_prompt_part2 = {
      'zh': f"根据以下描述来扮演{self.role_info.bot_chat}和{self.role_info.user_chat}对话，在对话中加入描述角色的感情、想法、身体动作等内容，也可以加入对环境、场面或动作产生结果的描述，以此来促进对话的进展，这些描述要合理且文采斐然。\n",
      'en': f"The following is another coherent verbose detailed conversation between {self.role_info.user_chat} and {self.role_info.bot_chat}.\n"
    }
    init_prompt_final = init_prompt[self.lang]
    init_prompt_part2_final = init_prompt_part2[self.lang]
    if em:
      init_prompt_final += f'\n\n{em}\n\n{init_prompt_part2_final}'
    else:
      init_prompt_final = f'{init_prompt_part2_final}'
    init_prompt_final += f"{self.role_info.bot_persona}"
    init_prompt_final = init_prompt_final.strip().split('\n')
    for c in range(len(init_prompt_final)):
      init_prompt_final[c] = init_prompt_final[c].strip().strip('\u3000').strip('\r')
    init_prompt_final = '\n'.join(init_prompt_final).strip() + '\n\n'
    if self.role_info.greeting:
      init_prompt_final += f"{self.role_info.bot}: {self.role_info.greeting}\n\n"
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
    out, model_tokens, model_state = self.__get_init_state()
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
    for c in chatbot:
      if c[1]:
        i = c[1].replace(self.role_info.user_chat, '').replace(self.role_info.bot_chat, '')
        bot_token = self.model_utils.pipeline.encode(i)
        bot_token.reverse()
        for t in bot_token:
          if t in self.model_utils.AVOID_REPEAT_TOKENS:
            continue
          for o in occurrence:
            if occurrence[o] > 1:
              occurrence[o] *= 0.999
          occurrence[t] = 1 + (occurrence[t] if t in occurrence else 0)
    return occurrence