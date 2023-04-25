import os, json
import gradio as gr
from modules.model_utils import ModelUtils
from modules.chat import Chat
class UI:

  model_utils = None
  chat_model = None
  char_path = './chars'
  config_role_path = './config/config_role.json'
  lock_flag_role = True

  def __init__(self, model_utils:ModelUtils):
    self.model_utils = model_utils
    self.chat_model = Chat(model_utils)

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
  def __save_char(self, user='', bot='', greeting='', bot_persona=''):
    with open(f"{self.char_path}/{bot}.json", 'w', encoding='utf8') as f:
      json.dump({'user': user, 'bot': bot, 'greeting': greeting, 'bot_persona': bot_persona}, f, indent=2, ensure_ascii=False)
    char_list = self.__get_json_files(self.char_path)
    return gr.Dropdown.update(choices=char_list)

  # 载入角色
  def __load_char(self, file_name):
    if not file_name:
      raise gr.Error("请选择一个角色")
    with open(f"{self.char_path}/{file_name}.json", 'r', encoding='utf-8') as f:
      char = json.loads(f.read())
    chatbot = self.chat_model.load_init_prompt(char['user'], char['bot'], char['greeting'], char['bot_persona'])
    return_arr = (
      char['user'], 
      char['bot'], 
      char['greeting'], 
      char['bot_persona'],
      chatbot, 
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

  def __send_message(self, message, top_p, top_k, temperature, presence_penalty, frequency_penalty):
    text, chatbot = self.chat_model.on_message(message, top_p, top_k, temperature, presence_penalty, frequency_penalty)
    show_label = False
    interactive = True
    if self.chat_model.check_token_count():
      show_label = True
      interactive = False
    result = (
      text,
      chatbot,
      gr.Textbox.update(show_label=show_label),
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
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True), 
      gr.Button.update(interactive=True)
    )
    return result


  def __reset_chatbot(self):
    message, chatbot = self.chat_model.reset_bot()
    return_arr = (
      message,
      chatbot,
      gr.Button.update(visible=True),
      gr.Button.update(visible=False),
      gr.Button.update(visible=False)
    )
    return return_arr
  
  def __unlock_param(self, flag):
    text = '锁定'
    if not flag:
      text = '解锁'
    return_arr = (
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
    with gr.Blocks(title="RWKV角色扮演") as app:
      if not os.path.isfile(self.config_role_path):
        self.__save_config_role()

      with gr.Tab("聊天"):
        with gr.Row():
          with gr.Column(scale=3):
            chatbot = gr.HTML(value=f'<style>{self.chat_model.chat_css}</style><div class="chat" id="chat"></div>')
            message = gr.Textbox(placeholder='说些什么吧', show_label=False, label='整理Token中……', interactive=False)
            with gr.Row():
              with gr.Column(min_width=100):
                submit = gr.Button('提交', interactive=False)       
              with gr.Column(min_width=100):
                get_prompt_btn = gr.Button('提词', interactive=False)
            with gr.Row():
              with gr.Column(min_width=100):
                regen = gr.Button('重新生成', interactive=False)
              with gr.Column(min_width=100):
                clear_last_btn = gr.Button('清除上一条', interactive=False)
            delete = gr.Button('清空聊天', interactive=False)
            with gr.Row():
              with gr.Column(min_width=100):
                clear_chat = gr.Button('清空', visible=False, elem_classes='warn_btn')
              with gr.Column(min_width=100):
                clear_cancel = gr.Button('取消', visible=False)
          with gr.Column(scale=1):
            with gr.Row():
              char_dropdown = gr.Dropdown(None, interactive=True, label="请选择角色")
            with gr.Row():
              with gr.Column(min_width=100):
                refresh_char_btn = gr.Button("刷新角色列表")
              with gr.Column(min_width=100):
                load_char_btn = gr.Button("载入角色")
            top_p = gr.Slider(minimum=0, maximum=1.0, step=0.01, interactive=False, label='Top P')
            top_k = gr.Slider(minimum=0, maximum=200, step=1, interactive=False, label='Top K')
            temperature = gr.Slider(minimum=0.2, maximum=5.0, step=0.01, interactive=False, label='Temperature')
            presence_penalty = gr.Slider(minimum=0, maximum=1.0, step=0.01, interactive=False, label='Presence Penalty')
            frequency_penalty = gr.Slider(minimum=0, maximum=1.0, step=0.01, interactive=False, label='Frequency Penalty')
            with gr.Row():
              with gr.Column(min_width=100):
                unlock_btn = gr.Button('解锁')
              with gr.Column(min_width=100):
                save_conf = gr.Button('保存')

      with gr.Tab("角色"):
        with gr.Row():
          with gr.Column():
            user = gr.Textbox(placeholder='AI怎么称呼你', label='你的名字')
            bot = gr.Textbox(placeholder='角色名字', label='角色的名字')
            greeting = gr.Textbox(placeholder='开场白', label='开场白')
          with gr.Column():
            bot_persona = gr.TextArea(placeholder='角色性格', label='角色的性格', lines=10)
        save_char_btn = gr.Button('保存角色')
      
      input_list = [message, top_p, top_k, temperature, presence_penalty, frequency_penalty]
      output_list = [message, chatbot]
      char_input_list = [user, bot, greeting, bot_persona, chatbot]
      interactive_list = [message, submit, regen, delete, clear_last_btn, get_prompt_btn]

      load_char_btn.click(self.__load_char, inputs=[char_dropdown], outputs=char_input_list + interactive_list)
      refresh_char_btn.click(self.__update_chars_list, outputs=[char_dropdown])
      save_conf.click(self.__save_config_role, inputs=input_list[1:])
      message.submit(self.__send_message, inputs=input_list, outputs=output_list + interactive_list).then(self.__arrange_token, outputs=interactive_list, show_progress=False)
      submit.click(self.__send_message, inputs=input_list, outputs=output_list + interactive_list).then(self.__arrange_token, outputs=interactive_list, show_progress=False)
      regen.click(self.chat_model.regen_msg, inputs=input_list[1:], outputs=output_list)
      save_char_btn.click(self.__save_char, inputs=char_input_list[:-1], outputs=[char_dropdown])
      clear_last_btn.click(self.chat_model.clear_last, outputs=[chatbot, message])
      get_prompt_btn.click(self.chat_model.get_prompt, inputs=input_list[1:], outputs=[message])
      unlock_btn.click(self.__unlock_role_param, outputs=input_list[1:] + [unlock_btn])
      clear_chat.click(self.__reset_chatbot, outputs=output_list + [delete, clear_chat, clear_cancel])
      delete.click(self.__confirm_delete, outputs=[delete, clear_chat, clear_cancel])
      clear_cancel.click(self.__confirm_cancel, outputs=[delete, clear_chat, clear_cancel])

      with gr.Tab('调试'):
        test_now = gr.TextArea(placeholder='当前token', label='输出')
        test_pre = gr.TextArea(placeholder='上一次token', label='输出')
        test_btn = gr.Button('查看token')
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