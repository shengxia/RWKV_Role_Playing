pip install torch --extra-index-url https://download.pytorch.org/whl/cu117 --upgrade
pip install -r requirements.txt

python webui.py --model model/path

example: python webui.py --model model/fp16i8_RWKV-4-Pile-7B-EngChn-test5-20230326

更多的参数可以参考modules/options.py里面的配置，虽然也没啥东西，如果运行不起来，把webui.py里面的os.environ["RWKV_CUDA_ON"] = '1'改成os.environ["RWKV_CUDA_ON"] = '0'