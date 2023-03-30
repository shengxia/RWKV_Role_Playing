import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

from modules.model import load_model
from modules.options import cmd_opts
from modules.ui import create_ui

if __name__ == "__main__":
  load_model()
  create_ui().launch(server_name="0.0.0.0" if cmd_opts.listen else None, server_port=cmd_opts.port)