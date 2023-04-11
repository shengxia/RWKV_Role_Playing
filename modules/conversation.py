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
    interface = ":"
    user = "Bob"
    bot = "Alice"
    init_prompt = f'''
    The following is a coherent verbose detailed conversation between a Chinese girl named {bot} and her friend {user}. \
    {bot} is very intelligent, creative and friendly. \
    {bot} likes to tell {user} a lot about herself and her opinions. \
    {bot} usually gives {user} kind, helpful and informative advices.

    {user}{interface} lhc

    {bot}{interface} LHC是指大型强子对撞机（Large Hadron Collider），是世界最大最强的粒子加速器，由欧洲核子中心（CERN）在瑞士日内瓦地下建造。LHC的原理是加速质子（氢离子）并让它们相撞，让科学家研究基本粒子和它们之间的相互作用，并在2012年证实了希格斯玻色子的存在。
    
    {user}{interface} 企鹅会飞吗
    
    {bot}{interface} 企鹅是不会飞的。企鹅的翅膀短而扁平，更像是游泳时的一对桨。企鹅的身体结构和羽毛密度也更适合在水中游泳，而不是飞行。
    '''
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
      init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + f"\n\n{bot}{interface} "
    out, self.model_tokens, self.model_state = self.model_utils.run_rnn(self.model_tokens, self.model_state, self.model_utils.pipeline.encode(init_prompt))
    self.model_utils.save_all_stat(self.srv_con, 'con_init', out, self.model_tokens, self.model_state)
    self.model_utils.save_all_stat(self.srv_con, 'con', out, self.model_tokens, self.model_state)
    gc.collect()
    torch.cuda.empty_cache()
  
  def on_message_con(self, message_con, chatbot_con, top_p_con, temperature_con, presence_penalty_con, frequency_penalty_con):
    msg = message_con.replace('\\n','\n').strip()
    interface = ":"
    user = "Bob"
    bot = "Alice"
    new = f"{user}{interface} {msg}\n\n{bot}{interface} "
    chatbot_con = chatbot_con + [[msg, None]]
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_con, 'con')
    out, self.model_tokens, self.model_state = self.model_utils.run_rnn(self.model_tokens, self.model_state, self.model_utils.pipeline.encode(new), newline_adj=-999999999)
    self.model_utils.save_all_stat(self.srv_con, 'con_pre', out, self.model_tokens, self.model_state)
    new_reply, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature_con, top_p_con, presence_penalty_con, frequency_penalty_con)
    self.model_utils.save_all_stat(self.srv_con, 'con', out, self.model_tokens, self.model_state)
    chatbot_con[-1][1] = new_reply.replace('\n', '')
    return '', chatbot_con

  def regen_msg_con(self, chatbot_con, top_p_con, temperature_con, presence_penalty_con, frequency_penalty_con):
    try:
      out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_con, 'con_pre')
    except:
      return chatbot_con
    new_reply, out, self.model_tokens, self.model_state = self.model_utils.get_reply(self.model_tokens, self.model_state, out, temperature_con, top_p_con, presence_penalty_con, frequency_penalty_con)
    self.model_utils.save_all_stat(self.srv_con, 'con', out, self.model_tokens, self.model_state)
    chatbot_con[-1][1] = new_reply.replace('\n', '')
    return chatbot_con

  def reset_con(self):
    out, self.model_tokens, self.model_state = self.model_utils.load_all_stat(self.srv_con, 'con_init')
    self.model_utils.save_all_stat(self.srv_con, 'con', out, self.model_tokens, self.model_state)
    return None, []