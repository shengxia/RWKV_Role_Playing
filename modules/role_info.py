class RoleInfo:

  def __init__(self, file_name, chatbot, user, bot, greeting, bot_persona, example_message, use_qa, log_hash):
    self.file_name = file_name
    self.chatbot = chatbot
    self.user_chat = user
    self.bot_chat = bot
    self.user = user if not use_qa else user + '|User'
    self.bot = bot if not use_qa else bot + '|Assistant' 
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
          bot.append([self.remove_qa_prefix(current_msg),self.remove_qa_prefix(next_msg)])
        else:
          bot.append([self.remove_qa_prefix(current_msg),None])
      else:
          bot.append([None,self.remove_qa_prefix(current_msg)])
    return bot


  
  def is_user(self,msg:str)->bool:
    return msg.startswith("<user>") or msg.startswith(self.user_chat)

  def remove_qa_prefix(self,msg:str)->str:
    return (msg.
            removeprefix("<user>:").
            removeprefix("<bot>:").
            removeprefix(f"{self.bot}:").
            removeprefix(f"{self.user}:").
            removeprefix(f"{self.bot_chat}:").
            removeprefix(f"{self.user_chat}:").
            strip()
            )