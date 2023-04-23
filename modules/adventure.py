from modules.model_utils import ModelUtils

class Adventure:
    
  model_utils = None
  srv_adv = 'adv_server'

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils

  def load_background(self, chatbot, top_p, top_k, temperature, presence_penalty, frequency_penalty, background):
    model_tokens = []
    model_state = None
    init_prompt = f"{self.model_utils.user}: {background}\n\n{self.model_utils.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat(self.srv_adv, 'adv_init', out, model_tokens, model_state)
    self.model_utils.save_all_stat(self.srv_adv, 'adv_pre', out, model_tokens, model_state)
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    chatbot = [[None, new_reply.replace('\n', '')]]
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, model_tokens, model_state)
    return chatbot
  
  def on_message_adv(self, message, chatbot, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    new = f"{self.model_utils.user}: {message}\n\n{self.model_utils.bot}:"
    chatbot = chatbot + [[message, None]]
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv')
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new))
    self.model_utils.save_all_stat(self.srv_adv, 'adv_pre', out, model_tokens, model_state)
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, model_tokens, model_state)
    chatbot[-1][1] = new_reply.replace('\n', '')
    return '', chatbot

  def regen_msg_adv(self, chatbot, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv_pre')
    except:
      return chatbot
    chat_param = self.model_utils.format_chat_param(top_p, top_k, temperature, presence_penalty, frequency_penalty)
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, chat_param)
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, model_tokens, model_state)
    chatbot[-1][1] = new_reply.replace('\n', '')
    return chatbot

  def reset_adv(self):
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv_init')
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, model_tokens, model_state)
    return None, []