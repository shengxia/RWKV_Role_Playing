import re

class RoleInfo:

  def __init__(self, file_name, chatbot, user, bot, greeting, bot_persona, example_message, log_hash):
    self.file_name = file_name
    self.chatbot = chatbot
    self.user = user
    self.bot = bot
    self.greeting_raw = greeting
    self.greeting_chatbot = self.parse_greeting()
    if greeting:
      self.chatbot = self.greeting_chatbot.copy()
    self.bot_persona = bot_persona
    self.example_message = example_message
    self.log_hash = log_hash

  def parse_greeting(self)->list[list[str]]:
    if not self.greeting_raw:
      return ''
    greetinglist = self.greeting_raw.split('\n\n')
    chatbot = []
    self.greeting = ''
    while greetinglist:
      current_msg = greetinglist.pop(0)
      if self.is_user(current_msg):
        if greetinglist and not self.is_user(greetinglist[0]):
          next_msg = greetinglist.pop(0)
          u = {'char': self.user, 'msg': self.remove_qa_prefix(current_msg)}
          b = {'char': self.bot, 'msg': self.remove_qa_prefix(next_msg)}
          chatbot.append([u,b])
          self.greeting += f"Input: {u['msg']}\n\n"
          self.greeting += f"Response: {b['msg']}\n\n"
        else:
          u = {'char': self.bot, 'msg': self.remove_qa_prefix(current_msg)}
          chatbot.append([u,None])
          self.greeting += f"Input: {u['msg']}\n\n"
      else:
        b = {'char': self.bot, 'msg': self.remove_qa_prefix(current_msg)}
        chatbot.append([None, b])
        self.greeting += f"Response: {b['msg']}\n\n"
    return chatbot
  
  def is_user(self,msg:str)->bool:
    return msg.startswith("{{user}}:") or msg.startswith(f"{self.user}:")

  def remove_qa_prefix(self,msg:str)->str:
    return (msg.
            removeprefix("{{user}}:").
            removeprefix("{{char}}:").
            removeprefix(f"{self.bot}:").
            removeprefix(f"{self.user}:").
            strip().
            replace("{{user}}", self.user).
            replace("{{char}}", self.bot)
            )