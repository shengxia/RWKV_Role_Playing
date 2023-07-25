class RoleInfo:

  def __init__(self, file_name, chatbot, user, bot, action_start, action_end, greeting, bot_persona, example_message, use_qa, log_hash):
    self.file_name = file_name
    self.chatbot = chatbot
    self.user_chat = user
    self.bot_chat = bot
    self.user = user if not use_qa else 'User'
    self.bot = bot if not use_qa else 'Assistant' 
    self.action_start = action_start 
    self.action_start_token = None
    self.action_end = action_end 
    self.action_end_token = None
    self.greeting = greeting 
    self.bot_persona = bot_persona
    self.example_message = example_message
    self.log_hash = log_hash 