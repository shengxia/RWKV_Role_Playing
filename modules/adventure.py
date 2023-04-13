from modules.model_utils import ModelUtils
import gc, torch

class Adventure:
    
  model_utils = None
  model_tokens = []
  model_state = None
  srv_adv = 'adv_server'

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils

  def load_background(self, chatbot_adv, top_p_adv, temperature_adv, presence_penalty_adv, frequency_penalty_adv, background_adv, max_token):
    self.model_tokens = []
    self.model_state = None
    init_prompt = self.model_utils.get_default_prompt(background_adv)
    out, self.model_tokens, self.model_state = self.model_utils.run_rnn(self.model_tokens, self.model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat(self.srv_adv, 'adv_init', out, self.model_tokens, self.model_state)
    new_reply, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature_adv, top_p_adv, presence_penalty_adv, frequency_penalty_adv, max_token=max_token)
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, self.model_tokens, self.model_state)
    gc.collect()
    torch.cuda.empty_cache()
    chatbot_adv = [[None, new_reply.replace('\n', '')]]
    return chatbot_adv
  
  def on_message_adv(self, message_adv, chatbot_adv, top_p_adv, temperature_adv, presence_penalty_adv, frequency_penalty_adv, max_token):
    msg = message_adv.replace('\\n','\n').strip()
    new = f" {msg}\n{self.model_utils.bot}: "
    chatbot_adv = chatbot_adv + [[msg, None]]
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv')
    out, self.model_tokens, self.model_state = self.model_utils.run_rnn(self.model_tokens, self.model_state, self.model_utils.pipeline.encode(new), newline_adj=-999999999)
    self.model_utils.save_all_stat(self.srv_adv, 'adv_pre', out, self.model_tokens, self.model_state)
    new_reply, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature_adv, top_p_adv, presence_penalty_adv, frequency_penalty_adv, max_token=max_token)
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, self.model_tokens, self.model_state)
    chatbot_adv[-1][1] = new_reply.replace('\n', '')
    return '', chatbot_adv

  def regen_msg_adv(self, chatbot_adv, top_p_adv, temperature_adv, presence_penalty_adv, frequency_penalty_adv, max_token):
    try:
      out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv_pre')
    except:
      return chatbot_adv
    new_reply, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature_adv, top_p_adv, presence_penalty_adv, frequency_penalty_adv, max_token=max_token)
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, self.model_tokens, self.model_state)
    chatbot_adv[-1][1] = new_reply.replace('\n', '')
    return chatbot_adv

  def reset_adv(self):
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv_init')
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, self.model_tokens, self.model_state)
    return None, []