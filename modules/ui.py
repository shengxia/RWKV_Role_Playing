import os, json
import gradio as gr
from modules.model import on_message
from modules.model import reset_bot
from modules.model import regen_msg
from modules.model import load_init_prompt
from modules.model import get_prompt

def get_all_chars():
  files=os.listdir('./chars')
  char_list = []
  for f in files:
    file_name_arr = f.split('.')
    if file_name_arr[1] == 'json':
      char_list.append(file_name_arr[0])
  return char_list

def update_chars_list():
  char_list = get_all_chars()
  return gr.Dropdown.update(choices=char_list)

def save_config(top_p=0.7, temperature=2, presence_penalty=0.5, frequency_penalty=0.5):
  with open('config.json', 'w', encoding='utf8') as f:
    json.dump({'top_p': top_p, 'temperature': temperature, 'presence': presence_penalty, 'frequency': frequency_penalty}, f, indent=2)

def save_char(user='', bot='', greeting='', bot_persona='', scenario='', example_dialogue='', chatbot=[]):
  with open('./chars/' + bot + '.json', 'w', encoding='utf8') as f:
    json.dump({'user': user, 'bot': bot, 'greeting': greeting, 'bot_persona': bot_persona, 'scenario': scenario, 'example_dialogue': example_dialogue}, f, indent=2, ensure_ascii=False)
  return load_init_prompt(user, bot, greeting, bot_persona, scenario, example_dialogue, chatbot)

def load_config(file_name):
  with open('config.json', 'r', encoding='utf-8') as f:
    configs = json.loads(f.read())
    
  with open('./chars/' + file_name + '.json', 'r', encoding='utf-8') as f:
    char = json.loads(f.read())
  load_init_prompt(char['user'], char['bot'], char['greeting'], char['bot_persona'], char['scenario'], char['example_dialogue'], [])
  chatbot = [[None, char['greeting']]]
  return configs['top_p'], configs['temperature'], configs['presence'], configs['frequency'], char['user'], char['bot'], char['greeting'], char['bot_persona'], char['scenario'], char['example_dialogue'], chatbot

def clear_last(chatbot):
  message = chatbot[-1][0]
  if(len(chatbot) < 2):
    return chatbot, message
  chatbot = chatbot[0:-1]
  return chatbot, message

def create_ui():
  with gr.Blocks(title="RWKV角色扮演") as app:
    char_list = get_all_chars()
    if not os.path.isfile('config.json'):
      save_config()

    with gr.Tab("聊天"):
      with gr.Row():
        with gr.Column(scale=3):
          top_p = gr.Slider(minimum=0, maximum=1.0, step=0.01, label='Top P')
          temperature = gr.Slider(minimum=0.2, maximum=5.0, step=0.01, label='Temperature')
          presence_penalty = gr.Slider(minimum=0, maximum=1, step=0.01, label='Presence Penalty')
          frequency_penalty = gr.Slider(minimum=0, maximum=1, step=0.01, label='Frequency Penalty')
          save_conf = gr.Button('保存设置')
          with gr.Row():
            char_dropdown = gr.Dropdown(char_list, value=char_list[0], interactive=True, label="请选择角色")
          refresh_char_btn = gr.Button("刷新角色列表")
          load_char_btn = gr.Button("载入角色")
        with gr.Column(scale=7):
          chatbot = gr.Chatbot(show_label=False).style(height=380)
          message = gr.Textbox(placeholder='说些什么吧', show_label=False)
          with gr.Row():
            submit = gr.Button('提交')
            get_prompt_btn = gr.Button('提词')
          with gr.Row():
            regen = gr.Button('重新生成')
            clear_last_btn = gr.Button('清除上一条')
          delete = gr.Button('清空聊天')
    
    with gr.Tab("角色"):
      with gr.Row():
        with gr.Column(scale=5):
          user = gr.Textbox(placeholder='AI怎么称呼你', label='你的名字')
          bot = gr.Textbox(placeholder='角色名字', label='角色的名字')
          greeting = gr.Textbox(placeholder='开场白', label='开场白')
        with gr.Column(scale=5):
          bot_persona = gr.TextArea(placeholder='角色性格', label='角色的性格', lines=10)
      with gr.Row():
        with gr.Column(scale=5):
          scenario = gr.TextArea(placeholder='对话发生在什么背景下', label='背景故事', lines=10)
        with gr.Column(scale=5):
          example_dialogue = gr.TextArea(placeholder='示例对话', label='示例对话', lines=10)
      save_char_btn = gr.Button('保存角色')

    input_list = [message, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot]
    output_list = [message, chatbot]
    char_input_list = [user, bot, greeting, bot_persona, scenario, example_dialogue, chatbot]
    reload_list = [top_p, temperature, presence_penalty, frequency_penalty, user, bot, greeting, bot_persona, scenario, example_dialogue, chatbot]

    app.load(load_config, inputs=[char_dropdown], outputs=reload_list)
    load_char_btn.click(load_config, inputs=[char_dropdown], outputs=reload_list)
    refresh_char_btn.click(update_chars_list, inputs=None, outputs=[char_dropdown])
    save_conf.click(save_config, inputs=input_list[2:6])
    message.submit(on_message, inputs=input_list, outputs=output_list)
    submit.click(on_message, inputs=input_list, outputs=output_list)
    regen.click(regen_msg, inputs=input_list[1:6], outputs=output_list)
    delete.click(reset_bot, inputs=[greeting], outputs=output_list)
    save_char_btn.click(save_char, inputs=char_input_list, outputs=char_input_list)
    clear_last_btn.click(clear_last, inputs=[chatbot], outputs=[chatbot, message])
    get_prompt_btn.click(get_prompt, inputs=input_list[2:7], outputs=[message])
  return app