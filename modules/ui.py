import os, json
import gradio as gr
from modules.model import on_message
from modules.model import reset_bot
from modules.model import regen_msg
from modules.model import load_init_prompt

def save_config(top_p=0.7, temperature=0.95, presence_penalty=0.2, frequency_penalty=0.2):
  with open('config.json', 'w', encoding='utf8') as f:
    json.dump({'top_p': top_p, 'temperature': temperature, 'presence': presence_penalty, 'frequency': frequency_penalty}, f, indent=2)

def save_char(user='', bot='', greeting='', bot_persona='', scenario='', example_dialogue='', chatbot=[]):
  with open('char.json', 'w', encoding='utf8') as f:
    json.dump({'user': user, 'bot': bot, 'greeting': greeting, 'bot_persona': bot_persona, 'scenario': scenario, 'example_dialogue': example_dialogue}, f, indent=2, ensure_ascii=False)
  return load_init_prompt(user, bot, greeting, bot_persona, scenario, example_dialogue, chatbot)

def load_config():
  print('reload')
  with open('config.json', 'r', encoding='utf-8') as f:
    configs = json.loads(f.read())
    
  with open('char.json', 'r', encoding='utf-8') as f:
    char = json.loads(f.read())
  load_init_prompt(char['user'], char['bot'], char['greeting'], char['bot_persona'], char['scenario'], char['example_dialogue'], [])
  chatbot = [[None, char['greeting']]]
  return configs['top_p'], configs['temperature'], configs['presence'], configs['frequency'], char['user'], char['bot'], char['greeting'], char['bot_persona'], char['scenario'], char['example_dialogue'], chatbot

def create_ui():
  with gr.Blocks() as demo:
    if not os.path.isfile('config.json'):
      save_config()
      
    if not os.path.isfile('char.json'):
      save_char()

    gr.Markdown('''<h1><center>ChatRWKV WebUI</center></h1>''')

    with gr.Row():
      top_p = gr.Slider(minimum=0, maximum=1.0, step=0.01, label='Top P')
      temperature = gr.Slider(minimum=0.2, maximum=5.0, step=0.01, label='Temperature')
      presence_penalty = gr.Slider(minimum=0, maximum=1, step=0.01, label='Presence Penalty')
      frequency_penalty = gr.Slider(minimum=0, maximum=1, step=0.01, label='Frequency Penalty')
    save_conf = gr.Button('保存设置')

    chatbot = gr.Chatbot(show_label=False)
    message = gr.Textbox(placeholder='说些什么吧', show_label=False)

    with gr.Row():
      submit = gr.Button('提交')
      regen = gr.Button('重新生成')

    delete = gr.Button('清空聊天')

    user = gr.Textbox(placeholder='AI怎么称呼你', label='你的名字')
    bot = gr.Textbox(placeholder='角色名字', label='角色的名字')
    greeting = gr.Textbox(placeholder='开场白', label='开场白')
    bot_persona = gr.TextArea(placeholder='角色性格', label='角色的性格')
    scenario = gr.TextArea(placeholder='背景故事', label='对话发生在什么背景下')
    example_dialogue = gr.TextArea(placeholder='示例对话', label='示例对话')
    save_char_btn = gr.Button('保存并载入角色')

    input_list = [message, chatbot, top_p, temperature, presence_penalty, frequency_penalty, user, bot]
    output_list = [message, chatbot]
    char_input_list = [user, bot, greeting, bot_persona, scenario, example_dialogue, chatbot]
    reload_list = [top_p, temperature, presence_penalty, frequency_penalty, user, bot, greeting, bot_persona, scenario, example_dialogue, chatbot]

    demo.load(load_config, inputs=None, outputs=reload_list)
    save_conf.click(save_config, inputs=input_list[2:6])
    message.submit(on_message, inputs=input_list, outputs=output_list)
    submit.click(on_message, inputs=input_list, outputs=output_list)
    regen.click(regen_msg, inputs=input_list[1:6], outputs=output_list)
    delete.click(reset_bot, inputs=[greeting], outputs=output_list)
    save_char_btn.click(save_char, inputs=char_input_list, outputs=char_input_list)
  return demo