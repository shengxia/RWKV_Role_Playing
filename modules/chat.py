from modules.model_utils import ModelUtils
from pathlib import Path
import os, gc, json, datetime
import torch
import pickle
import time
import copy

class Chat:
  
  model_utils = None
  log_name = ''
  srv_chat = 'chat_server'
  chat_css = ''
  chatbot = []

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils
    with open('./css/chat.css', 'r') as f:
      self.chat_css = f.read()

  def load_init_prompt(self, user, bot, greeting, bot_persona):
    self.log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    model_tokens = []
    model_state = None
    if os.path.exists(f'save/{bot}.sav'):
      data = self.load_chat(bot)
      self.model_utils.save_all_stat(self.srv_chat, 'chat', data['out'], data['model_tokens'], data['model_state'])
      self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', data['out_pre'], data['model_tokens_pre'], data['model_state_pre'])
      self.model_utils.save_all_stat('', 'chat_init', data['out_init'], data['model_tokens_init'], data['model_state_init'])
      self.chatbot = data['chatbot']
    else:
      init_prompt = f"你是{bot}，{bot_persona}，{bot}称呼我为{user}。\n"
      init_prompt += f"{bot}: 我的名字叫{bot}，你叫什么名字？（感到好奇）\n"
      init_prompt += f"{user}: 你可以叫我{user}。\n"
      init_prompt += f"{bot}: 你好啊，{user}，很高兴认识你。（得到了回答，开心地伸出了手想要和{user}握手）\n"
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
      self.chatbot = [[None, greeting]]
      self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    gc.collect()
    torch.cuda.empty_cache()
    return self.generate_cai_chat_html(user, bot)
  
  def reset_bot(self, greeting, user, bot):
    self.log_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json'
    out, model_tokens, model_state = self.model_utils.load_all_stat('', 'chat_init')
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    self.chatbot = [[None, greeting]]
    return None, self.generate_cai_chat_html(user, bot)
  
  def regen_msg(self, top_p, temperature, presence_penalty, frequency_penalty, user, bot):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    except:
      return '', self.chatbot
    return self.gen_msg(out, top_p, temperature, presence_penalty, frequency_penalty, user, bot, model_tokens, model_state)
  
  def on_message(self, message, top_p, temperature, presence_penalty, frequency_penalty, user, bot):
    msg = message.replace('\\n','\n').strip()
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    new = f" {msg}\n{bot}: "
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new), newline_adj=-999999999)
    self.model_utils.save_all_stat(self.srv_chat, 'chat_pre', out, model_tokens, model_state)
    self.chatbot += [[msg, None]]
    return self.gen_msg(out, top_p, temperature, presence_penalty, frequency_penalty, user, bot, model_tokens, model_state) 
  
  def gen_msg(self, out, top_p, temperature, presence_penalty, frequency_penalty, user, bot, model_tokens, model_state):
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, temperature, top_p, presence_penalty, frequency_penalty, user, bot)
    self.model_utils.save_all_stat(self.srv_chat, 'chat', out, model_tokens, model_state)
    self.chatbot[-1][1] = new_reply.replace('\n', '')
    self.save_log()
    self.save_chat(bot)
    return '', self.generate_cai_chat_html(user, bot)

  def save_log(self):
    os.makedirs('log', exist_ok=True)
    dict_list = [{'input': q, 'output': a} for q, a in self.chatbot]
    with open(f'log/{self.log_name}', 'w', encoding='utf-8') as f:
      json.dump(dict_list, f, ensure_ascii=False, indent=2)

  def get_prompt(self, top_p, temperature, presence_penalty, frequency_penalty, user, bot):
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    new_prompt = self.model_utils.get_reply(model_tokens, model_state, out, temperature, top_p, presence_penalty, frequency_penalty, user, bot)
    return new_prompt.replace('\n', '')
  
  def clear_last(self, user, bot):
    message = self.chatbot[-1][0]
    if(len(self.chatbot) < 2):
      return self.generate_cai_chat_html(user, bot), message
    self.chatbot = self.chatbot[0:-1]
    save_file = f'save/{bot}.sav'
    if os.path.exists(save_file):
      os.remove(save_file)
    return self.generate_cai_chat_html(user, bot), message
  
  def save_chat(self, bot):
    os.makedirs('save', exist_ok=True)
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_chat, 'chat')
    out_pre, model_tokens_pre, model_state_pre = self.model_utils.load_all_stat(self.srv_chat, 'chat_pre')
    out_init, model_tokens_init, model_state_init = self.model_utils.load_all_stat('', 'chat_init')
    data = {
      "out": out,
      "model_tokens": model_tokens,
      "model_state": model_state,
      "out_pre": out_pre,
      "model_tokens_pre": model_tokens_pre,
      "model_state_pre": model_state_pre,
      "out_init": out_init,
      "model_tokens_init": model_tokens_init,
      "model_state_init": model_state_init,
      "chatbot": self.chatbot
    }
    with open(f'save/{bot}.sav', 'wb') as f:
      pickle.dump(data, f)

  def load_chat(self, bot):
    with open(f'save/{bot}.sav', 'rb') as f:
      data = pickle.load(f)
    return data
  
  def generate_cai_chat_html(self, user, bot):
    output = f'<style>{self.chat_css}</style><div class="chat" id="chat">'

    # We use ?name2 and ?time.time() to force the browser to reset caches
    img_bot = f'<img src="file/cache/pfp_character.png?{bot}">' if Path("cache/pfp_character.png").exists() else ''
    img_me = f'<img src="file/cache/pfp_me.png">' if Path("cache/pfp_me.png").exists() else ''

    chatbot = copy.deepcopy(self.chatbot)
    chatbot.reverse()
    for row in chatbot:
      output += f"""
        <div class="message">
          <div class="circle-bot">
            {img_bot}
          </div>
          <div class="text">
            <div class="username">
              {bot}
            </div>
            <div class="message-body">
              {row[1]}
            </div>
          </div>
        </div>
      """
      if row[0] != None:  # don't display empty user messages
        output += f"""
          <div class="message">
            <div class="circle-you">
              {img_me}
            </div>
            <div class="text">
              <div class="username">
                {user}
              </div>
              <div class="message-body">
                {row[0]}
              </div>
            </div>
          </div>
        """
    output += "</div>"
    return output