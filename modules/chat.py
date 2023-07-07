from modules.model_utils import ModelUtils
from pathlib import Path
import os, json, pickle, copy, re, uuid, random

class Chat:
  
  model_utils = None
  chat_css = ''
  muti_user = False

  def __init__(self, model_utils:ModelUtils, muti_user):
    self.model_utils = model_utils
    self.muti_user = muti_user
    with open('./css/chat.css', 'r') as f:
      self.chat_css = f.read()

  def load_init_prompt(self, all_state, user, bot, action_start, action_end, greeting, bot_persona, example_message, use_qa, as_default=False):
    model_tokens = []
    model_state = None
    all_state = {
      'chatbot': [],
      'user_chat': user,
      'bot_chat': bot,
      'user': user if not use_qa else 'User',
      'bot': bot if not use_qa else 'Assistant',
      'action_start': action_start,
      'action_end': action_end,
      'greeting': greeting,
      'log_hash': str(uuid.uuid1()).replace('-', '')
    }
    init_prompt = self.__get_init_prompt(all_state, bot, bot_persona, user, example_message, as_default)
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
      init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n'.join(init_prompt).strip() + '\n\n'
    if greeting:
      init_prompt += f"{all_state['bot']}: {greeting}\n\n"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(init_prompt))
    all_state = self.model_utils.save_all_stat(all_state, 'chat_init', out, model_tokens, model_state)
    if not self.muti_user and os.path.exists(f'save/{bot}.sav'):
      data = self.__load_chat(all_state)
      all_state = self.model_utils.save_all_stat(all_state, 'chat', data['out'], data['model_tokens'], data['model_state'])
      if data['model_tokens_pre']:
        all_state = self.model_utils.save_all_stat(all_state, 'chat_pre', data['out_pre'], data['model_tokens_pre'], data['model_state_pre'])
      all_state['chatbot'] = data['chatbot']
    else:
      if greeting:
        all_state['chatbot'] = [[None, greeting]]
      all_state = self.model_utils.save_all_stat(all_state, 'chat', out, model_tokens, model_state)
    return self.__generate_cai_chat_html(all_state), all_state
  
  def reset_bot(self, all_state):
    out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_init')
    all_state['log_hash'] = str(uuid.uuid1()).replace('-', '')
    all_state = self.model_utils.save_all_stat(all_state, 'chat', out, model_tokens, model_state)
    try:
      all_state = self.model_utils.remove_stat(all_state, 'chat_pre')
    except:
      pass
    if all_state["greeting"]:
      all_state["chatbot"] = [[None, all_state["greeting"]]]
    else:
      all_state["chatbot"] = []
    save_file = f'save/{all_state["bot_chat"]}.sav'
    if os.path.exists(save_file):
      os.remove(save_file)
    return None, None, self.__generate_cai_chat_html(all_state), all_state
  
  def regen_msg(self, all_state, top_p, temperature, presence_penalty, frequency_penalty, min_len):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_pre')
    except:
      return '', self.__generate_cai_chat_html(all_state)
    new = f"{all_state['user']}: {all_state['chatbot'][-1][0]}\n\n{all_state['bot']}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(
      top_p, temperature, presence_penalty, frequency_penalty, 
      min_len, all_state['action_start_token'], all_state['action_end_token']
    )
    occurrence = self.__get_occurrence(all_state, True)
    reply_text, all_state = self.gen_msg(all_state, out, chat_param, model_tokens, model_state, occurrence) 
    return '', '', reply_text, all_state
  
  def on_message(self, all_state, message, action, top_p, temperature, presence_penalty, frequency_penalty, action_front, min_len, replace_message):
    message = message.strip().replace('\r\n','\n') if message else ''
    action = action.strip().replace('\r\n','\n') if action else ''
    msg = f"{message}"
    if action_front:
      if action:
        msg = f"{all_state['action_start']}{action}{all_state['action_end']}{msg}"
    else:
      if action:
        msg += f"{all_state['action_start']}{action}{all_state['action_end']}"
    if replace_message:
      try:
        out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_pre')
      except:
        return '', '', self.__generate_cai_chat_html(all_state)
      new = f"{all_state['user']}: {all_state['chatbot'][-1][0]}\n\n{all_state['bot']}: {msg}\n\n"
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
      all_state['chatbot'][-1][1] = msg
      all_state = self.model_utils.save_all_stat(all_state, 'chat', out, model_tokens, model_state)
      return '', '', self.__generate_cai_chat_html(all_state), all_state
    else:
      out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat')
      all_state = self.model_utils.save_all_stat(all_state, 'chat_pre', out, model_tokens, model_state)
      new = f"{all_state['user']}: "
      new += f"{msg}\n\n{all_state['bot']}:"
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
      all_state['chatbot'] += [[msg, None]]
      chat_param = self.model_utils.format_chat_param(
        top_p, temperature, presence_penalty, frequency_penalty, 
        min_len, all_state['action_start_token'], all_state['action_end_token']
      )
      occurrence = self.__get_occurrence(all_state)
      reply_text, all_state = self.gen_msg(all_state, out, chat_param, model_tokens, model_state, occurrence)
      return '', '', reply_text, all_state
  
  def gen_msg(self, all_state, out, chat_param, model_tokens, model_state, occurrence):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param, occurrence)
    all_state['chatbot'][-1][1] = new_reply
    all_state = self.model_utils.save_all_stat(all_state, 'chat', out, model_tokens, model_state)
    self.__save_log(all_state)
    self.__save_chat(all_state)
    return self.__generate_cai_chat_html(all_state), all_state
    
  def get_prompt(self, all_state, top_p, temperature, presence_penalty, frequency_penalty):
    out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat')
    new = f"{all_state['user']}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    chat_param = self.model_utils.format_chat_param(top_p, temperature, presence_penalty, frequency_penalty)
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    pos_arr = list(self.__find_all_chat(new_prompt[0], all_state))
    chat_action_data = self.__format_chat_action(pos_arr, new_prompt[0])
    chat, action = self.__get_chat_action(chat_action_data)
    return chat, action
  
  def clear_last(self, all_state):
    n = 1
    if(len(all_state['chatbot']) == 0):
      return self.__generate_cai_chat_html(all_state), '', ''
    if not all_state['chatbot'][0][0]:
      n += 1
      if(len(all_state['chatbot']) == 1):
        return self.__generate_cai_chat_html(all_state), '', ''
    messages = all_state['chatbot'].pop()    
    if len(all_state['chatbot']) < n:
      out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_init')
      all_state = self.model_utils.save_all_stat(all_state, 'chat', out, model_tokens, model_state)
      all_state = self.model_utils.remove_stat(all_state, 'chat_pre')
    elif len(all_state['chatbot']) < n + 1:
      out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_pre')
      all_state = self.model_utils.save_all_stat(all_state, 'chat', out, model_tokens, model_state)
      out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_init')
      all_state = self.model_utils.save_all_stat(all_state, 'chat_pre', out, model_tokens, model_state)
    else:
      out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_pre')
      all_state = self.model_utils.save_all_stat(all_state, 'chat', out, model_tokens, model_state)
      chat_str = self.__get_chatbot_str(all_state['chatbot'][1:-1], all_state)
      out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_init')
      out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
      all_state = self.model_utils.save_all_stat(all_state, 'chat_pre', out, model_tokens, model_state)
    self.__save_chat(all_state)
    self.__save_log(all_state)
    pos_arr = list(self.__find_all_chat(messages[0], all_state))
    chat_action_data = self.__format_chat_action(pos_arr, messages[0])
    chat, action = self.__get_chat_action(chat_action_data)
    return self.__generate_cai_chat_html(all_state), chat, action, all_state
  
  def __save_log(self, all_state):
    os.makedirs(f'log/{all_state["bot_chat"]}/', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in all_state['chatbot']]
    with open(f'./log/{all_state["bot_chat"]}/{all_state["log_hash"]}.json', 'w', encoding='utf-8') as f:
      json.dump(dict_list, f, ensure_ascii=False, indent=2)

  def __save_chat(self, all_state):
    if self.muti_user:
      return
    os.makedirs('save', exist_ok=True)
    out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat')
    try:
      out_pre, model_tokens_pre, model_state_pre = self.model_utils.load_all_stat(all_state, 'chat_pre')
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
      "chatbot": all_state['chatbot']
    }
    with open(f'save/{all_state["bot_chat"]}.sav', 'wb') as f:
      pickle.dump(data, f)

  def __load_chat(self, all_state):
    with open(f'save/{all_state["bot_chat"]}.sav', 'rb') as f:
      data = pickle.load(f)
    return data
  
  def __generate_cai_chat_html(self, all_state):
    output = f'<style>{self.chat_css}</style><div class="chat" id="chat">'
    img_bot = f'<img src="file/chars/{all_state["bot_chat"]}.png">' if Path(f'chars/{all_state["bot_chat"]}.png').exists() else ''
    chatbot = copy.deepcopy(all_state['chatbot'])
    chatbot.reverse()
    for row in chatbot:
      pos_arr = list(self.__find_all_chat(row[1], all_state))
      chat_action_data = self.__format_chat_action(pos_arr, row[1])
      msg = self.__format_chat_html(chat_action_data).replace('\n', '<br>')
      output += f"""
        <div class="message message_c">
          <div class="circle-bot">
            {img_bot}
          </div>
          <div class="text_c">
            <div class="username">
              {all_state['bot_chat']}
            </div>
            <div class="message-body message-body-c">
              {msg}
            </div>
          </div>
        </div>
      """
      if row[0] != None:
        pos_arr = list(self.__find_all_chat(row[0], all_state))
        chat_action_data = self.__format_chat_action(pos_arr, row[0])
        msg = self.__format_chat_html(chat_action_data).replace('\n', '<br>')
        output += f"""
          <div class="message message_m">
            <div class="text_m">
              <div class="username username-m">
                {all_state['user_chat']}
              </div>
              <div class="message-body message-body-m">
                {msg}
              </div>
            </div>
          </div>
        """
    output += "</div>"
    return output.replace('<br><br>', '<br>').replace('<br><br>', '<br>')
  
  def __get_chatbot_str(self, chatbot, all_state):
    chat_str = ''
    for row in chatbot:
      chat_str += f'{all_state["user"]}: {row[0]}\n\n'
      chat_str += f'{all_state["bot"]}: {row[1]}\n\n'
    return chat_str
  
  # 目前这里我准备写成随机生成初始prompt的方法，我也不知道为什么，多次重复载入同样的初始prompt，生成效果会变差（对比测试一下）
  def __get_init_prompt(self, all_state, bot, bot_persona, user, example_message, as_default=False):
    if not as_default:
      if all_state['action_start'] and all_state['action_start'] in example_message and all_state['action_end'] in example_message:
        all_state['action_start_token'] = self.model_utils.pipeline.encode(f' {all_state["action_start"]}')
        all_state['action_end_token'] = self.model_utils.pipeline.encode(all_state['action_end'])
      else:
        all_state['action_start_token'] = None
        all_state['action_end_token'] = None
      em = example_message.replace('<bot>', bot).replace('<user>', user)
      init_prompt = [
        f"阅读并理解以下{user}和{bot}之间的对话：",
        f"The following is a coherent verbose detailed conversation between {user} and {bot}."
      ]
      init_prompt_part2 = [
        f"根据以下描述来扮演{bot}和我对话，在对话中加入描述角色的感情、想法、身体动作等内容，也可以加入对环境、场面或动作产生结果的描述，以此来促进对话的进展，这些描述要合理且文采斐然。\n",
        f"The following is another coherent verbose detailed conversation between {user} and {bot}.\n"
      ]
      # init_prompt_final = init_prompt[random.randint(0, 1)]
      # init_prompt_part2_final = init_prompt_part2[random.randint(0, 1)]
      init_prompt_final = init_prompt[0]
      init_prompt_part2_final = init_prompt_part2[0]
      if em:
        init_prompt_final += f'\n\n{em}\n\n{init_prompt_part2_final}'
      else:
        init_prompt_final = f'{init_prompt_part2_final}'
      init_prompt_final += f"{bot_persona}"
    else:
      init_prompt_final = "User: hi\n\nAssistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
    return init_prompt_final

  def get_test_data(self, all_state):
    data_now = self.model_utils.load_all_stat(all_state, 'chat')
    txt_now = f"token count: {len(data_now[1])}\n\n{self.model_utils.pipeline.decode(data_now[1])}"
    try:
      data_pre = self.model_utils.load_all_stat(all_state, 'chat_pre')
      txt_pre = f"token count: {len(data_pre[1])}\n\n{self.model_utils.pipeline.decode(data_pre[1])}"
    except:
      txt_pre = ''
    return txt_now, txt_pre
  
  def check_token_count(self, all_state):
    data = self.model_utils.load_all_stat(all_state, 'chat')
    if len(data[1]) < 4000:
      return False
    return True

  def arrange_token(self, all_state):
    out, model_tokens, model_state = self.model_utils.load_all_stat(all_state, 'chat_init')
    chat_str = ''
    chat_str_pre = ''
    i = 0
    for row in reversed(all_state['chatbot'][:-1]):
      if len(chat_str_pre) > 400:
        break
      chat_str_pre = f'{all_state["bot"]}: {row[1]}\n\n' + chat_str_pre
      chat_str_pre = f'{all_state["user"]}: {row[0]}\n\n' + chat_str_pre
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str_pre))
    all_state = self.model_utils.save_all_stat(all_state, 'chat_pre', out, model_tokens, model_state)
    chat_str += f'{all_state["user"]}: {all_state["chatbot"][-1][0]}\n\n'
    chat_str += f'{all_state["bot"]}: {all_state["chatbot"][-1][1]}\n\n'
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(chat_str))
    all_state = self.model_utils.save_all_stat(all_state, 'chat', out, model_tokens, model_state)
    return all_state

  def __find_all_chat(self, input_str, all_state):
    if not all_state['action_start'] or not all_state['action_end']:
      return (0, len(input_str))
    pattern = re.compile("\\" + all_state['action_start'] + ".*?\\" + all_state['action_end'])
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
  
  def __get_occurrence(self, all_state, is_pre=False):
    chatbot = copy.deepcopy(all_state['chatbot'])
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