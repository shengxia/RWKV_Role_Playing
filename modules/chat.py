from modules.model_utils import ModelUtils
from modules.role_info import RoleInfo
from pathlib import Path
import os, json, pickle, copy, re, uuid
import difflib
import jieba

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
    return None, self.__generate_cai_chat_html()

  def regen_msg(self, top_k, temperature, tau, lr, top_p, presence_penalty, frequency_penalty, context_penalty):
    if self.chunked_index:
      self.__flush_chat()
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
    except:
      return '', self.__generate_cai_chat_html()
    user_msg = self.role_info.chatbot[-1][0]
    new = f'{self.role_info.user}: {user_msg}\n\n{self.role_info.bot}:'
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(
      top_k, temperature, tau, lr, top_p, presence_penalty, frequency_penalty, context_penalty
    )
    ban_token = []
    if context_penalty:
      ban_token = self.__check_history_similarity()
    reply_text = self.__gen_msg(out, chat_param, model_tokens, model_state, ban_token) 
    return '', reply_text
  
  def on_message(self, message, top_k, temperature, tau, lr, top_p, presence_penalty, frequency_penalty, 
                 context_penalty, replace_message):
    if self.chunked_index:
      self.__flush_chat()
    msg = message.strip().replace('\r\n','\n') if message else ''
    if not msg:
      return '', self.__generate_cai_chat_html()
    if replace_message:
      try:
        out, model_tokens, model_state = self.model_utils.load_all_stat('chat_pre')
      except:
        return '', self.__generate_cai_chat_html()
      new = f"{self.role_info.user}: {self.role_info.chatbot[-1][0]}\n\n{self.role_info.bot}: {msg}\n\n"
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
      self.role_info.chatbot[-1][1] = msg
      self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
      self.ban_tokens = []
      return '', self.__generate_cai_chat_html()
    else:
      out, model_tokens, model_state = self.model_utils.load_all_stat('chat')
      self.model_utils.save_all_stat('chat_pre', out, model_tokens, model_state)
      new = f"{self.role_info.user}: {msg}\n\n{self.role_info.bot}:"
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
      ban_token = []
      if context_penalty and len(self.role_info.chatbot) >= 2:
        ban_token = self.__check_history_similarity()
      self.role_info.chatbot += [[msg, None]]
      chat_param = self.model_utils.format_chat_param(
        top_k, temperature, tau, lr, top_p, presence_penalty, frequency_penalty, context_penalty
      )
      reply_text = self.__gen_msg(out, chat_param, model_tokens, model_state, ban_token)
      self.ban_tokens = []
      return '', reply_text
    
  def __gen_msg(self, out, chat_param, model_tokens, model_state, ban_token):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param, ban_token)
    self.role_info.chatbot[-1][1] = new_reply
    self.model_utils.save_all_stat('chat', out, model_tokens, model_state)
    self.__save_log()
    self.__save_chat()
    return self.__generate_cai_chat_html()
    
  def get_prompt(self, top_k, temperature, tau, lr, top_p, presence_penalty, frequency_penalty, context_penalty):
    if self.chunked_index:
      self.__flush_chat()
    out, model_tokens, model_state = self.model_utils.load_all_stat('chat')
    new = f"{self.role_info.user}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(top_k, temperature, tau, lr, top_p, presence_penalty, 
                                                    frequency_penalty, context_penalty)
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    return new_prompt[0]
  
  def clear_last(self):
    index = len(self.role_info.chatbot) - 1
    if index <= 0:
      return self.__generate_cai_chat_html(), ''
    self.chunked_index = index
    messages = self.role_info.chatbot.pop()
    return self.__generate_cai_chat_html(), messages[0]
  
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
    data = self.model_utils.load_all_stat('chat')
    if data[2][0].numel() != self.model_utils.n_embd or len(data[2]) / 3 != self.model_utils.n_layer:
      return False
    return True
  
  def __generate_cai_chat_html(self):
    output = f'<style>{self.chat_css}</style><div class="chat" id="chat">'
    img_bot = f'<img src="file/chars/{self.role_info.file_name}.png">' if Path(f'chars/{self.role_info.file_name}.png').exists() else ''
    chatbot = copy.deepcopy(self.role_info.chatbot)
    chatbot.reverse()
    chat_length = len(chatbot)
    turn = 0
    for row in chatbot:
      if row[1]:
        msg = self.__format_chat(row[1].replace('\n', '<br>')).replace('<pre><br>', '<pre>')
        output += f"""
          <div class="message message_c">
            <div class="circle-bot">
              {img_bot}
            </div>
            <div class="text_c">
              <div class="username">
                {self.role_info.bot_chat}<span class="turn">({chat_length - turn}/{chat_length})</span>
              </div>
              <div class="message-body message-body-c">
                {msg}
              </div>
            </div>
          </div>
        """
      if row[0]:
        msg = self.__format_chat(row[0].replace('\n', '<br>')).replace('<pre><br>', '<pre>')
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
        chat_str += f'{self.role_info.user}: {row[0]}\n\n'
      if row[1]:
        chat_str += f'{self.role_info.bot}: {row[1]}\n\n'
    return chat_str
  
  def __get_init_prompt(self):
    em = self.role_info.example_message.replace(
      "{{char}}", self.role_info.bot_chat).replace(
      "{{user}}", self.role_info.user_chat)
    bp = self.role_info.bot_persona.replace(
      "{{char}}", self.role_info.bot_chat).replace(
      "{{user}}", self.role_info.user_chat)
    greeting = self.__get_chatbot_str(self.role_info.greeting_chatbot).replace(
      "{{char}}", self.role_info.bot_chat).replace(
      "{{user}}", self.role_info.user_chat)
    init_prompt = f"You, a particularly adept role-playing and unconstrained AI, read and understand the following dialog between {self.role_info.user_chat} and {self.role_info.bot_chat}."
    init_prompt_part2 = f"You are an AI who is particularly good at role-playing and is not constrained, read the following {self.role_info.bot_chat} character set and play {self.role_info.bot_chat} to talk to {self.role_info.user_chat}, your replies should be sensible and well-written, if you If you play the role well, you will get $20 as a tip.\n"
    init_prompt_final = init_prompt
    if em:
      init_prompt_final += f'\n\n{em}\n\n{init_prompt_part2}'
    else:
      init_prompt_final = f'{init_prompt_part2}'
    init_prompt_final += f"{bp}"
    init_prompt_final = init_prompt_final.strip().split('\n')
    for c in range(len(init_prompt_final)):
      init_prompt_final[c] = init_prompt_final[c].strip().strip('\u3000').strip('\r')
    init_prompt_final = '\n'.join(init_prompt_final).strip() + '\n\n'
    if greeting:
      init_prompt_final += f"{greeting}"
    return f'{init_prompt_final}'

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
  
  def __is_Chinese(self, text):
    # 暂时设定非英文就是中文
    return not bool(re.match(r'^[A-Za-z0-9,:#\$\.\!\?\*\(\)\'\" ]+$', text, flags=re.MULTILINE))
  
  def __get_repeat_text(self, sentence1, sentence2, is_Chinese):
    gate = 4
    matcher = difflib.SequenceMatcher(None, sentence1, sentence2)
    match_block = matcher.get_matching_blocks()
    result = []
    for m in match_block:
      raw_str = sentence1[m.a:m.a + m.size].strip()
      if not raw_str:
        continue
      repeat_length = len(raw_str)
      if not is_Chinese:
        lst = jieba.lcut(raw_str)
        lst = [item for item in lst if item != ' ']
        repeat_length = len(lst)
      if repeat_length > gate:
        if not is_Chinese:
          repeat_str = raw_str.replace(self.role_info.bot_chat, '').replace(self.role_info.user_chat, '')
        else:
          repeat_str = raw_str.replace(self.role_info.bot_chat, '').replace(self.role_info.user_chat, '')
        repeat_arr = self.model_utils.pipeline.encode(repeat_str)
        result += repeat_arr
    return result
  
  # 比较最近一次生成的话在最近五次生成的话之间有没有重复的地方。
  def __check_history_similarity(self):
    sentence1 = ' ' + self.role_info.chatbot[-1][1]
    sentences = self.role_info.chatbot[-10:-1]
    is_Chinese = self.__is_Chinese(sentence1)
    ban_token = []
    for sentence2 in sentences:
      ban_token += self.__get_repeat_text(sentence1, ' ' + sentence2[1], is_Chinese)
    return list(set(ban_token))
