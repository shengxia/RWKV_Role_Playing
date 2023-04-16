from modules.model_utils import ModelUtils
import gc, torch

class Adventure:
    
  model_utils = None
  srv_adv = 'adv_server'

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils

  def load_background(self, chatbot_adv, top_p_adv, temperature_adv, presence_penalty_adv, frequency_penalty_adv, background_adv):
    model_tokens = []
    model_state = None
    init_prompt = f"{self.model_utils.user}:{background_adv}\n{self.model_utils.bot}:"
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat(self.srv_adv, 'adv_init', out, model_tokens, model_state)
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, temperature_adv, top_p_adv, presence_penalty_adv, frequency_penalty_adv)
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, model_tokens, model_state)
    gc.collect()
    torch.cuda.empty_cache()
    chatbot_adv = [[None, new_reply.replace('\n', '')]]
    return chatbot_adv
  
  def on_message_adv(self, message_adv, chatbot_adv, top_p_adv, temperature_adv, presence_penalty_adv, frequency_penalty_adv):
    msg = message_adv.replace('\\n','\n').strip()
    new = f" {msg}\n{self.model_utils.bot}: "
    chatbot_adv = chatbot_adv + [[msg, None]]
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv')
    out, model_tokens, model_state = self.model_utils.run_rnn(model_tokens, model_state, self.model_utils.pipeline.encode(new), newline_adj=-999999999)
    self.model_utils.save_all_stat(self.srv_adv, 'adv_pre', out, model_tokens, model_state)
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, temperature_adv, top_p_adv, presence_penalty_adv, frequency_penalty_adv)
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, model_tokens, model_state)
    chatbot_adv[-1][1] = new_reply.replace('\n', '')
    return '', chatbot_adv

  def regen_msg_adv(self, chatbot_adv, top_p_adv, temperature_adv, presence_penalty_adv, frequency_penalty_adv):
    try:
      out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv_pre')
    except:
      return chatbot_adv
    new_reply, out, model_tokens, model_state = self.model_utils.get_reply(model_tokens, model_state, out, temperature_adv, top_p_adv, presence_penalty_adv, frequency_penalty_adv)
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, model_tokens, model_state)
    chatbot_adv[-1][1] = new_reply.replace('\n', '')
    return chatbot_adv

  def reset_adv(self):
    out, model_tokens, model_state = self.model_utils.load_all_stat(self.srv_adv, 'adv_init')
    self.model_utils.save_all_stat(self.srv_adv, 'adv', out, model_tokens, model_state)
    return None, []