import os, json
import time
import gradio as gr
from modules.model_utils import ModelUtils
from modules.chat import Chat
class UI:

  model_utils = None
  chat_model = None
  char_path = './chars'
  save_path = './save'
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
    file_list = self.__get_file_list_by_extend(path, "json")
    return file_list

  def __get_save_files(self, path):
    file_list = self.__get_file_list_by_extend(path, "sav")
    return file_list

  def __get_file_list_by_extend(self, path:str, file_extend:str):
      file_list = []
      if os.path.exists(path) and os.path.isdir(path):
        files=os.listdir(path)
        files.sort()
        for f in files:
          file_name_arr = f.split('.')
          if file_name_arr[-1] == file_extend:
            file_list.append(file_name_arr[0])
      return file_list

  # 更新角色列表
  def __update_chars_list(self):
    char_list = self.__get_json_files(self.char_path)
    return gr.Dropdown.update(choices=char_list)
  
  # 更新对话记录列表
  def __update_save_list(self, bot_name):
    save_list = self.__get_save_list(bot_name)
    return gr.Dropdown.update(choices=save_list)

  def __get_save_list(self, bot_name):
      save_list = [ f'{bot_name}/' + i for i in self.__get_save_files(f'{self.save_path}/{bot_name}')]
      if os.path.exists(f'{self.save_path}/{bot_name}.sav'):
        save_list.append(f'{bot_name}')
      return save_list
  
  def __save_config(self, f, top_p, tau, temperature, presence_penalty, frequency_penalty, cfg, min_len):
    config = {
      'min_len': min_len,
      'top_p': top_p, 
      'tau': tau, 
      'temperature': temperature, 
      'presence': presence_penalty, 
      'frequency': frequency_penalty,
      'cfg': cfg
    }
    json.dump(config, f, indent=2)

  # 保存角色扮演模式的配置
  def __save_config_role(self, top_p=0.65, tau=0, temperature=2, presence_penalty=0.2, frequency_penalty=0.2, cfg=0, min_len=0):
    with open(self.config_role_path, 'w', encoding='utf8') as f:
      self.__save_config(f, top_p, tau, temperature, presence_penalty, frequency_penalty, cfg, min_len)
  
  # 保存角色
  def __save_char(self, file_name='', user='', bot='', action_start='', action_end='', greeting='', bot_persona='', example_message='', use_qa=False):
    if file_name == '' and bot != '':
      file_name = bot
    with open(f"{self.char_path}/{file_name}.json", 'w', encoding='utf8') as f:
      char = {
        'user': user, 
        'bot': bot, 
        'action_start': action_start,
        'action_end': action_end,
        'greeting': greeting, 
        'bot_persona': bot_persona, 
        'example_message': example_message,
        'use_qa': use_qa
      }
      json.dump(char, f, indent=2, ensure_ascii=False)
      init_save_file = f"./chars/init_state/{file_name}.sav"
      if os.path.exists(init_save_file):
        os.remove(init_save_file)
      save_file = f'save/{file_name}.sav'
      if os.path.exists(save_file):
        os.remove(save_file)
    chatbot = self.chat_model.load_init_prompt(file_name, char['user'], char['bot'], char['action_start'], 
                                               char['action_end'], char['greeting'], char['bot_persona'], 
                                               char['example_message'], char['use_qa'])
    char_list = self.__get_json_files(self.char_path)
    return gr.Dropdown.update(choices=char_list), chatbot

  # 载入角色
  def __load_char(self, file_name):
    if not file_name:
      raise gr.Error(self.language_conf['LOAD_CHAR_ERROR'])
    with open(f"{self.char_path}/{file_name}.json", 'r', encoding='utf-8') as f:
      char = json.loads(f.read())
    for key in ['user', 'bot', 'action_start', 'action_end', 'greeting', 'bot_persona', 'example_message', 'use_qa']:
      if key not in char.keys():
        if key == 'use_qa':
          char[key] = False
        else:
          char[key] = ''
    chatbot = self.chat_model.load_init_prompt(file_name, char['user'], char['bot'], char['action_start'], 
                                               char['action_end'], char['greeting'], char['bot_persona'], 
                                               char['example_message'], char['use_qa'])
    return_arr = (
      file_name,
      char['user'], 
      char['bot'], 
      char['action_start'], 
      char['action_end'], 
      char['greeting'], 
      char['bot_persona'],
      char['example_message'],
      char['use_qa'],
      chatbot,
      self.__update_save_list(file_name),
      gr.Textbox.update(interactive=True), 
      gr.Textbox.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True)
    )
    return return_arr

  def __load_save(self, file_name):
    return (self.chat_model.load_state(file_name),)

  def __save_save(self, bot, file_name):
    if file_name == '':
      file_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_path=f'{bot}/{file_name}'
    self.chat_model.save_chat_to(save_path)
    return (gr.Dropdown.update(choices=self.__get_save_list(bot),value=save_path),)

  def __save_update(self, bot, file_name):
    self.chat_model.save_chat_to(file_name)
    return (gr.Dropdown.update(choices=self.__get_save_list(bot),value=file_name),)

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

  def __send_message(self, message, action, top_p, tau, temperature, presence_penalty, frequency_penalty, cfg, min_len, action_front, replace_message):
    text, action_text, chatbot = self.chat_model.on_message(message, action, top_p, tau, temperature, presence_penalty, frequency_penalty, cfg, action_front, min_len, replace_message)
    show_label = False
    interactive = True
    if self.chat_model.check_token_count():
      show_label = True
      interactive = False
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
      gr.Button.update(interactive=interactive),
      gr.Checkbox.update(value=False)
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
    
  # 初始化UI
  def __init_ui(self):
    with open(self.config_role_path, 'r', encoding='utf-8') as f:
      configs_role = json.loads(f.read())
    char_list = self.__get_json_files(self.char_path)
    if 'tau' not in configs_role:
      configs_role['tau'] = 0    
    if 'cfg' not in configs_role:
      configs_role['cfg'] = 0
    if 'min_len' not in configs_role:
      configs_role['min_len'] = 0
    return_arr = (
      configs_role['min_len'],
      configs_role['top_p'], 
      configs_role['tau'], 
      configs_role['temperature'], 
      configs_role['presence'], 
      configs_role['frequency'], 
      configs_role['cfg'], 
      gr.Dropdown.update(choices=char_list)
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
            with gr.Row():
              action_front = gr.Checkbox(label=self.language_conf['AF_CK'], value=True)
              replace_message = gr.Checkbox(label=self.language_conf['TAMPER'])
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
            with gr.Tab(self.language_conf['TAB_DEFAULT']):
              with gr.Row():
                char_dropdown = gr.Dropdown(None, interactive=True, label=self.language_conf['CHAR_DP_LB'])
              with gr.Row():
                with gr.Column(min_width=100):
                  refresh_char_btn = gr.Button(self.language_conf['REFRESH_CHAR'])
                with gr.Column(min_width=100):
                  load_char_btn = gr.Button(self.language_conf['LOAD_CHAR'])
            with gr.Tab(self.language_conf['TAB_SAVE']):
              with gr.Row():
                save_dropdown = gr.Dropdown(None, interactive=True, label=self.language_conf['SELECT_SAVE'])
              with gr.Row():
                with gr.Column(min_width=100):
                  refresh_save_btn = gr.Button(self.language_conf['LOAD_SAVE_LIST'])
                with gr.Column(min_width=100):
                  load_save_btn = gr.Button(self.language_conf['LOAD_SAVE'])
              with gr.Row():
                save_file_name = gr.Textbox(label=self.language_conf['SAVE_FILE'])
              with gr.Row():
                with gr.Column(min_width=100):
                  save_update_btn = gr.Button(self.language_conf['UPDATE_STATE'])
                with gr.Column(min_width=100):
                  save_btn = gr.Button(self.language_conf['SAVE_STATE'])
            with gr.Tab(self.language_conf['TAB_CONFIG']):  
              min_len = gr.Slider(minimum=0, maximum=500, step=1, label=self.language_conf['MIN_LEN'])
              top_p = gr.Slider(minimum=0, maximum=1.0, step=0.01, label='Top P')
              tau = gr.Slider(minimum=0, maximum=1, step=0.01, label='TAU')
              temperature = gr.Slider(minimum=0.1, maximum=5.0, step=0.01, label='Temperature')
              presence_penalty = gr.Slider(minimum=0, maximum=5.0, step=0.01, label='Presence Penalty')
              frequency_penalty = gr.Slider(minimum=0, maximum=5.0, step=0.01, label='Frequency Penalty')
              cfg = gr.Slider(minimum=0, maximum=2.0, step=0.1, label='cfg factor')
              with gr.Row():
                with gr.Column():
                  save_conf = gr.Button(self.language_conf['SAVE_CFG'])

      with gr.Tab(self.language_conf['CHAR']):
        with gr.Row():
          with gr.Column():
            file_name = gr.Textbox(placeholder=self.language_conf['FILE_NAME_PH'], label=self.language_conf['FILE_NAME_LB'])
            user = gr.Textbox(placeholder=self.language_conf['USER_PH'], label=self.language_conf['USER_LB'])
            bot = gr.Textbox(placeholder=self.language_conf['BOT_PH'], label=self.language_conf['BOT_LB'])
            use_qa = gr.Checkbox(label=self.language_conf['QA_REPLACE'])
            with gr.Row():
              with gr.Column(min_width=100):
                action_start = gr.Textbox(placeholder=self.language_conf['AC_START_LB'], label=self.language_conf['AC_START_LB'])
              with gr.Column(min_width=100):
                action_end = gr.Textbox(placeholder=self.language_conf['AC_END_LB'], label=self.language_conf['AC_END_LB'])
          with gr.Column():
            greeting = gr.TextArea(placeholder=self.language_conf['GREETING_PH'], label=self.language_conf['GREETING_LB'], lines=2)
            bot_persona = gr.TextArea(placeholder=self.language_conf['PERSONA_PH'], label=self.language_conf['PERSONA_LB'], lines=7)
        with gr.Row():
          example_message = gr.TextArea(placeholder=self.language_conf['EXAMPLE_DIA'], label=self.language_conf['EXAMPLE_DIA_LB'], lines=10)
        save_char_btn = gr.Button(self.language_conf['SAVE_CHAR'])
      
      input_list = [message, action, top_p, tau, temperature, presence_penalty, frequency_penalty, cfg, min_len]
      output_list = [message, action, chatbot]
      char_input_list = [file_name, user, bot, action_start, action_end, greeting, bot_persona, example_message, use_qa, chatbot]
      interactive_list = [message, action, submit, regen, delete, clear_last_btn, get_prompt_btn]

      load_char_btn.click(self.__load_char, inputs=[char_dropdown], outputs=char_input_list + [save_dropdown] + interactive_list)
      refresh_char_btn.click(self.__update_chars_list, outputs=[char_dropdown])
      refresh_save_btn.click(self.__update_save_list, inputs=[char_dropdown], outputs=[save_dropdown])
      load_save_btn.click(self.__load_save, inputs=[save_dropdown], outputs=[chatbot])
      save_btn.click(self.__save_save, inputs=[char_dropdown,save_file_name], outputs=[save_dropdown])
      save_update_btn.click(self.__save_update, inputs=[char_dropdown,save_dropdown], outputs=[save_dropdown])
      save_conf.click(self.__save_config_role, inputs=input_list[2:])
      message.submit(self.__send_message, inputs=input_list + [action_front, replace_message], outputs=output_list + interactive_list + [replace_message]).then(self.__arrange_token, outputs=interactive_list, show_progress=False)
      action.submit(self.__send_message, inputs=input_list + [action_front, replace_message], outputs=output_list + interactive_list + [replace_message]).then(self.__arrange_token, outputs=interactive_list, show_progress=False)
      submit.click(self.__send_message, inputs=input_list + [action_front, replace_message], outputs=output_list + interactive_list + [replace_message]).then(self.__arrange_token, outputs=interactive_list, show_progress=False)
      regen.click(self.chat_model.regen_msg, inputs=input_list[2:], outputs=output_list)
      save_char_btn.click(self.__save_char, inputs=char_input_list[:-1], outputs=[char_dropdown, chatbot])
      clear_last_btn.click(self.chat_model.clear_last, outputs=[chatbot, message, action])
      get_prompt_btn.click(self.chat_model.get_prompt, inputs=input_list[2:-2], outputs=[message, action])
      clear_chat.click(self.__reset_chatbot, outputs=output_list + [delete, clear_chat, clear_cancel])
      delete.click(self.__confirm_delete, outputs=[delete, clear_chat, clear_cancel])
      clear_cancel.click(self.__confirm_cancel, outputs=[delete, clear_chat, clear_cancel])

      with gr.Tab(self.language_conf['DEBUG']):
        test_now = gr.TextArea(placeholder=self.language_conf['TOKEN_NOW'], label=self.language_conf['OUTPUT'])
        test_pre = gr.TextArea(placeholder=self.language_conf['TOKEN_LAST'], label=self.language_conf['OUTPUT'])
        test_btn = gr.Button(self.language_conf['GET_TOKEN'])
      test_btn.click(self.chat_model.get_test_data, outputs=[test_now, test_pre])

      reload_list = [
        min_len,
        top_p,
        tau, 
        temperature, 
        presence_penalty, 
        frequency_penalty, 
        cfg,
        char_dropdown
      ]
      app.load(self.__init_ui, outputs=reload_list)
    return app