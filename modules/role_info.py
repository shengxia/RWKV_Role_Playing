import re

class RoleInfo:

  def __init__(self, file_name, chatbot, user, bot, greeting, bot_persona, example_message, use_qa, log_hash):
    self.file_name = file_name
    self.chatbot = chatbot
    self.user_chat = user
    self.bot_chat = bot
    self.use_qa = use_qa
    self.user = f'{user}' if not use_qa else 'User'
    self.bot = f'{bot}' if not use_qa else f'Assistant' 
    self.greeting = greeting
    self.greeting_chatbot = self.parse_greeting(greeting)
    if greeting:
      self.chatbot = self.greeting_chatbot.copy()
    self.bot_persona = bot_persona
    self.example_message = example_message
    self.log_hash = log_hash

  def parse_greeting(self,greeting:str)->list[list[str]]:
    if not greeting:
      return []
    while greeting.find('\n\n\n')!= -1:
      greeting = greeting.replace('\n\n\n','\n\n')
    greetinglist = greeting.split('\n\n')
    bot = []
    while greetinglist:
      current_msg = greetinglist.pop(0)
      if self.is_user(current_msg):
        if greetinglist and not self.is_user(greetinglist[0]):
          next_msg = greetinglist.pop(0)
          u = {'char': self.user_chat, 'msg': self.remove_qa_prefix(current_msg)}
          b = {'char': self.bot_chat, 'msg': self.remove_qa_prefix(next_msg)}
          bot.append([u,b])
        else:
          u = {'char': self.user_chat, 'msg': self.remove_qa_prefix(current_msg)}
          bot.append([u,None])
      else:
        b = {'char': self.bot_chat, 'msg': self.remove_qa_prefix(current_msg)}
        bot.append([None, b])
    return bot
  
  def is_user(self,msg:str)->bool:
    return msg.startswith("{{user}}:") or msg.startswith(f"{self.user_chat}:")

  def remove_qa_prefix(self,msg:str)->str:
    return (msg.
            removeprefix("{{user}}:").
            removeprefix("{{char}}:").
            removeprefix(f"{self.bot}:").
            removeprefix(f"{self.user}:").
            removeprefix(f"{self.bot_chat}:").
            removeprefix(f"{self.user_chat}:").
            strip().
            replace("{{user}}", self.user_chat).
            replace("{{char}}", self.get_pure_char_name())
            )
  
  def get_pure_char_name(self):
    pattern = re.compile(r'-(.*)')
    char_name = re.sub(pattern, '', self.bot_chat)
    return char_name