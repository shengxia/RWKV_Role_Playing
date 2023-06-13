import os, json
import gradio as gr
from modules.model_utils import ModelUtils
from modules.chat import Chat
class UI:

  model_utils = None
  chat_model = None
  char_path = './chars'
  config_role_path = './config/config_role.json'
  language_path = './language/'
  lock_flag_role = True
  language_conf = None

  def __init__(self, model_utils:ModelUtils, lang):
    self.model_utils = model_utils
    self.chat_model = Chat(model_utils, lang)
    with open(f"{self.language_path}/{lang}.json", 'r', encoding='utf-8') as f:
      self.language_conf = json.loads(f.read())

  def __get_json_files(self, path):
    files=os.listdir(path)
    file_list = []
    for f in files:
      file_name_arr = f.split('.')
      if file_name_arr[-1] == 'json':
        file_list.append(file_name_arr[0])
    return file_list

  # 更新角色列表
  def __update_chars_list(self):
    char_list = self.__get_json_files(self.char_path)
    return gr.Dropdown.update(choices=char_list)
  
  def __save_config(self, f, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    config = {
      'top_p': top_p, 
      'top_k': top_k, 
      'temperature': temperature, 
      'presence': presence_penalty, 
      'frequency': frequency_penalty
    }
    json.dump(config, f, indent=2)

  # 保存角色扮演模式的配置
  def __save_config_role(self, top_p=0.7, top_k=0, temperature=2, presence_penalty=0.5, frequency_penalty=0.5):
    with open(self.config_role_path, 'w', encoding='utf8') as f:
      self.__save_config(f, top_p, top_k, temperature, presence_penalty, frequency_penalty)
  
  # 保存角色
  def __save_char(self, user='', bot='', action_start='', action_end='', greeting='', bot_persona='', example_message=''):
    with open(f"{self.char_path}/{bot}.json", 'w', encoding='utf8') as f:
      char = {
        'user': user, 
        'bot': bot, 
        'action_start': action_start,
        'action_end': action_end,
        'greeting': greeting, 
        'bot_persona': bot_persona, 
        'example_message': example_message
      }
      json.dump(char, f, indent=2, ensure_ascii=False)
    char_list = self.__get_json_files(self.char_path)
    return gr.Dropdown.update(choices=char_list)

  # 载入角色
  def __load_char(self, file_name):
    if not file_name:
      raise gr.Error(self.language_conf['LOAD_CHAR_ERROR'])
    with open(f"{self.char_path}/{file_name}.json", 'r', encoding='utf-8') as f:
      char = json.loads(f.read())
    for key in ['user', 'bot', 'action_start', 'action_end', 'greeting', 'bot_persona', 'example_message']:
      if key not in char.keys():
        char[key] = ''
    chatbot = self.chat_model.load_init_prompt(char['user'], char['bot'], char['action_start'], 
                                               char['action_end'], char['greeting'], char['bot_persona'], 
                                               char['example_message'])
    return_arr = (
      char['user'], 
      char['bot'], 
      char['action_start'], 
      char['action_end'], 
      char['greeting'], 
      char['bot_persona'],
      char['example_message'],
      chatbot, 
      gr.Textbox.update(interactive=True), 
      gr.Textbox.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True)
    )
    return return_arr

  def __confirm_delete(self):
    return_arr = (
      gr.Button.update(visible=False),
      gr.Button.update(visible=True),
      gr.Button.update(visible=True)
    )
    return return_arr
  
  def __confirm_cancel(self):
    return_arr = (
      gr.Button.update(visible=True),
      gr.Button.update(visible=False),
      gr.Button.update(visible=False)
    )
    return return_arr
  
  def __unlock_role_param(self):
    return_arr = self.__unlock_param(self.lock_flag_role)
    self.lock_flag_role = not self.lock_flag_role
    return return_arr

  def __send_message(self, message, action, top_p, top_k, temperature, presence_penalty, frequency_penalty, turns, min_len, action_front):
    text, action_text, chatbot = self.chat_model.on_message(message, action, top_p, top_k, temperature, presence_penalty, frequency_penalty, action_front, turns, min_len)
    show_label = False
    interactive = True
    # if self.chat_model.check_token_count():
    #   show_label = True
    #   interactive = False
    result = (
      text,
      action_text,
      chatbot,
      gr.Textbox.update(show_label=show_label),
      gr.Textbox.update(interactive=interactive), 
      gr.Button.update(interactive=interactive), 
      gr.Button.update(interactive=interactive), 
      gr.Button.update(interactive=interactive), 
      gr.Button.update(interactive=interactive), 
      gr.Button.update(interactive=interactive)
    )
    return result
  
  def __arrange_token(self):
    if self.chat_model.check_token_count():
      self.chat_model.arrange_token()
    result = (
      gr.Textbox.update(show_label=False),
      gr.Textbox.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True)
    )
    return result

  def __reset_chatbot(self):
    message, action, chatbot = self.chat_model.reset_bot()
    return_arr = (
      message,
      action,
      chatbot,
      gr.Button.update(visible=True),
      gr.Button.update(visible=False),
      gr.Button.update(visible=False)
    )
    return return_arr
  
  def __unlock_param(self, flag):
    text = self.language_conf['LOCK']
    if not flag:
      text = self.language_conf['UNLOCK']
    return_arr = (
      gr.Slider.update(interactive=flag), 
      gr.Slider.update(interactive=flag), 
      gr.Slider.update(interactive=flag), 
      gr.Slider.update(interactive=flag), 
      gr.Slider.update(interactive=flag), 
      gr.Slider.update(interactive=flag), 
      gr.Slider.update(interactive=flag), 
      gr.Button.update(value=text)
    )
    return return_arr
    
  # 初始化UI
  def __init_ui(self):
    with open(self.config_role_path, 'r', encoding='utf-8') as f:
      configs_role = json.loads(f.read())
    char_list = self.__get_json_files(self.char_path)
    return_arr = (
      configs_role['top_p'], 
      configs_role['top_k'], 
      configs_role['temperature'], 
      configs_role['presence'], 
      configs_role['frequency'], 
      gr.Dropdown.update(choices=char_list), 
    )
    return return_arr

  # 创建UI
  def create_ui(self):
    with gr.Blocks(title=self.language_conf['TITLE']) as app:
      if not os.path.isfile(self.config_role_path):
        self.__save_config_role()

      with gr.Tab(self.language_conf['CHAT_TAB']):
        with gr.Row():
          with gr.Column(scale=3):
            chatbot = gr.HTML(value=f'<style>{self.chat_model.chat_css}</style><div class="chat" id="chat"></div>')
            message = gr.Textbox(placeholder=self.language_conf['MSG_PH'], show_label=False, label=self.language_conf['MSG_LB'], interactive=False)
            action = gr.Textbox(placeholder=self.language_conf['NARR_PH'], show_label=False, interactive=False)
            action_front = gr.Checkbox(label=self.language_conf['AF_CK'],)
            with gr.Row():
              with gr.Column(min_width=150):
                submit = gr.Button(self.language_conf['SUBMIT'], interactive=False)       
              with gr.Column(min_width=150):
                regen = gr.Button(self.language_conf['REGEN'], interactive=False)
              with gr.Column(min_width=150):
                get_prompt_btn = gr.Button(self.language_conf['GET_PROMPT'], interactive=False)
              with gr.Column(min_width=150):
                clear_last_btn = gr.Button(self.language_conf['CLEAR_LAST'], interactive=False)
            delete = gr.Button(self.language_conf['CLEAR_CHAT'], interactive=False)
            with gr.Row():
              with gr.Column(min_width=100):
                clear_chat = gr.Button(self.language_conf['CLEAR'], visible=False, elem_classes='warn_btn')
              with gr.Column(min_width=100):
                clear_cancel = gr.Button(self.language_conf['CANCEL'], visible=False)
          with gr.Column(scale=1):
            with gr.Row():
              char_dropdown = gr.Dropdown(None, interactive=True, label=self.language_conf['CHAR_DP_LB'])
            with gr.Row():
              with gr.Column(min_width=100):
                refresh_char_btn = gr.Button(self.language_conf['REFRESH_CHAR'])
              with gr.Column(min_width=100):
                load_char_btn = gr.Button(self.language_conf['LOAD_CHAR'])
            turns = gr.Slider(minimum=1, maximum=10, step=1, interactive=False, label=self.language_conf['TURNS'])
            min_len = gr.Slider(minimum=0, maximum=500, step=1, interactive=False, label='最小回复长度（0为不控制）')
            top_p = gr.Slider(minimum=0, maximum=1.0, step=0.01, interactive=False, label='Top P')
            top_k = gr.Slider(minimum=0, maximum=200, step=1, interactive=False, label='Top K')
            temperature = gr.Slider(minimum=0.2, maximum=5.0, step=0.01, interactive=False, label='Temperature')
            presence_penalty = gr.Slider(minimum=0, maximum=1.0, step=0.01, interactive=False, label='Presence Penalty')
            frequency_penalty = gr.Slider(minimum=0, maximum=1.0, step=0.01, interactive=False, label='Frequency Penalty')
            with gr.Row():
              with gr.Column(min_width=100):
                unlock_btn = gr.Button(self.language_conf['UNLOCK'])
              with gr.Column(min_width=100):
                save_conf = gr.Button(self.language_conf['SAVE_CFG'])

      with gr.Tab(self.language_conf['CHAR']):
        with gr.Row():
          with gr.Column():
            user = gr.Textbox(placeholder=self.language_conf['USER_PH'], label=self.language_conf['USER_LB'])
            bot = gr.Textbox(placeholder=self.language_conf['BOT_PH'], label=self.language_conf['BOT_LB'])
            with gr.Row():
              with gr.Column(min_width=100):
                action_start = gr.Textbox(placeholder=self.language_conf['AC_START_LB'], label=self.language_conf['AC_START_LB'])
              with gr.Column(min_width=100):
                action_end = gr.Textbox(placeholder=self.language_conf['AC_END_LB'], label=self.language_conf['AC_END_LB'])
          with gr.Column():
            greeting = gr.TextArea(placeholder=self.language_conf['GREETING_PH'], label=self.language_conf['GREETING_LB'], lines=2)
            bot_persona = gr.TextArea(placeholder=self.language_conf['PERSONA_PH'], label=self.language_conf['PERSONA_LB'], lines=6)
        with gr.Row():
          example_message = gr.TextArea(placeholder=self.language_conf['EXAMPLE_DIA'], label=self.language_conf['EXAMPLE_DIA_LB'], lines=10)
        save_char_btn = gr.Button(self.language_conf['SAVE_CHAR'])
      
      input_list = [message, action, top_p, top_k, temperature, presence_penalty, frequency_penalty, turns, min_len]
      output_list = [message, action, chatbot]
      char_input_list = [user, bot, action_start, action_end, greeting, bot_persona, example_message, chatbot]
      interactive_list = [message, action, submit, regen, delete, clear_last_btn, get_prompt_btn]

      load_char_btn.click(self.__load_char, inputs=[char_dropdown], outputs=char_input_list + interactive_list)
      refresh_char_btn.click(self.__update_chars_list, outputs=[char_dropdown])
      save_conf.click(self.__save_config_role, inputs=input_list[2:-2])
      # message.submit(self.__send_message, inputs=input_list + [action_front], outputs=output_list + interactive_list).then(self.__arrange_token, outputs=interactive_list, show_progress=False)
      # action.submit(self.__send_message, inputs=input_list + [action_front], outputs=output_list + interactive_list).then(self.__arrange_token, outputs=interactive_list, show_progress=False)
      # submit.click(self.__send_message, inputs=input_list + [action_front], outputs=output_list + interactive_list).then(self.__arrange_token, outputs=interactive_list, show_progress=False)      
      message.submit(self.__send_message, inputs=input_list + [action_front], outputs=output_list + interactive_list)
      action.submit(self.__send_message, inputs=input_list + [action_front], outputs=output_list + interactive_list)
      submit.click(self.__send_message, inputs=input_list + [action_front], outputs=output_list + interactive_list)
      regen.click(self.chat_model.regen_msg, inputs=input_list[2:], outputs=output_list)
      save_char_btn.click(self.__save_char, inputs=char_input_list[:-1], outputs=[char_dropdown])
      clear_last_btn.click(self.chat_model.clear_last, outputs=[chatbot, message, action])
      get_prompt_btn.click(self.chat_model.get_prompt, inputs=input_list[2:-2], outputs=[message, action])
      unlock_btn.click(self.__unlock_role_param, outputs=input_list[2:] + [unlock_btn])
      clear_chat.click(self.__reset_chatbot, outputs=output_list + [delete, clear_chat, clear_cancel])
      delete.click(self.__confirm_delete, outputs=[delete, clear_chat, clear_cancel])
      clear_cancel.click(self.__confirm_cancel, outputs=[delete, clear_chat, clear_cancel])

      with gr.Tab(self.language_conf['DEBUG']):
        test_now = gr.TextArea(placeholder=self.language_conf['TOKEN_NOW'], label=self.language_conf['OUTPUT'])
        test_pre = gr.TextArea(placeholder=self.language_conf['TOKEN_LAST'], label=self.language_conf['OUTPUT'])
        test_btn = gr.Button(self.language_conf['GET_TOKEN'])
      test_btn.click(self.chat_model.get_test_data, outputs=[test_now, test_pre])

      reload_list = [
        top_p, 
        top_k, 
        temperature, 
        presence_penalty, 
        frequency_penalty, 
        char_dropdown
      ]
      app.load(self.__init_ui, outputs=reload_list)

    return app