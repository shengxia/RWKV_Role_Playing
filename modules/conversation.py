from modules.model_utils import ModelUtils
import gc, torch

class Conversation:
    
  model_utils = None
  model_tokens = []
  model_state = None
  srv_con = 'con_server'

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils

  def init_conversation(self):
    self.model_tokens = []
    self.model_state = None
    init_prompt = self.model_utils.get_default_prompt()
    out, self.model_tokens, self.model_state = self.model_utils.run_rnn(self.model_tokens, self.model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat(self.srv_con, 'con_init', out, self.model_tokens, self.model_state)
    self.model_utils.save_all_stat(self.srv_con, 'con', out, self.model_tokens, self.model_state)
    gc.collect()
    torch.cuda.empty_cache()
  
  def on_message_con(self, message_con, chatbot_con, top_p_con, temperature_con, presence_penalty_con, frequency_penalty_con, max_token):
    msg = message_con.replace('\\n','\n').strip()
    new = f" {msg}\n{self.model_utils.bot}: "
    chatbot_con = chatbot_con + [[msg, None]]
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_con, 'con')
    out, self.model_tokens, self.model_state = self.model_utils.run_rnn(self.model_tokens, self.model_state, self.model_utils.pipeline.encode(new), newline_adj=-999999999)
    self.model_utils.save_all_stat(self.srv_con, 'con_pre', out, self.model_tokens, self.model_state)
    new_reply, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature_con, top_p_con, presence_penalty_con, frequency_penalty_con, max_token=max_token)
    self.model_utils.save_all_stat(self.srv_con, 'con', out, self.model_tokens, self.model_state)
    chatbot_con[-1][1] = new_reply.replace('\n', '')
    return '', chatbot_con

  def regen_msg_con(self, chatbot_con, top_p_con, temperature_con, presence_penalty_con, frequency_penalty_con, max_token):
    try:
      out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_con, 'con_pre')
    except:
      return chatbot_con
    new_reply, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature_con, top_p_con, presence_penalty_con, frequency_penalty_con, max_token=max_token)
    self.model_utils.save_all_stat(self.srv_con, 'con', out, self.model_tokens, self.model_state)
    chatbot_con[-1][1] = new_reply.replace('\n', '')
    return chatbot_con

  def reset_con(self):
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_con, 'con_init')
    self.model_utils.save_all_stat(self.srv_con, 'con', out, self.model_tokens, self.model_state)
    return None, []