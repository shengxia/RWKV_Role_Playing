import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default="7860")
parser.add_argument("--model", type=str, default="model/fp16i8_RWKV-4-Raven-7B-v10x-Eng49-Other1%-20230423-ctx4096")
parser.add_argument("--strategy", type=str, default="cuda fp16i8 *20 -> cuda fp16")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--cuda_on", type=str, default="0", help="RWKV_CUDA_ON value")
parser.add_argument("--jit_on", type=str, default="1", help="RWKV_JIT_ON value")
parser.add_argument("--share", action="store_true", help="use gradio share")
parser.add_argument("--lang", type=str, default="zh", help="zh: Chinese; en: English")
cmd_opts = parser.parse_args()

import os
os.environ["RWKV_JIT_ON"] = cmd_opts.jit_on
os.environ["RWKV_CUDA_ON"] = cmd_opts.cuda_on

from modules.model_utils import ModelUtils
from modules.ui import UI

if __name__ == "__main__":
  model_util = ModelUtils(cmd_opts)
  model_util.load_model()
  ui = UI(model_util, cmd_opts.lang)
  app = ui.create_ui()
  app.queue(concurrency_count=5, max_size=64).launch(
    server_name="0.0.0.0" if cmd_opts.listen else None, 
    share=cmd_opts.share,
    server_port=cmd_opts.port
  )