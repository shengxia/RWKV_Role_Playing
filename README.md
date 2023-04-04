## 一个基于RWKV的角色扮演玩具

![图片1](./pic/1.png)
![图片2](./pic/2.png)

就是这么一个玩意儿，连抄带编的弄出来了一个玩具，所以代码质量吗……请各位不要吐槽太多，但是也算是能玩吧。

### 安装方法：

先安装依赖
```
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117 --upgrade

pip install -r requirements.txt
```

启动：
```
python webui.py --listen --model model/path
```

以下是一个例子: 
```
python webui.py --listen --model model/fp16i8_RWKV-4-Pile-7B-EngChn-test5-20230326
```

更多的参数可以参考modules/options.py里面的配置，模型的加载方式（--strategy）我默认使用的是"cuda fp16i8"，如果想使用其他的加载方式可以自行调整该参数，具体有哪些值可以参考[这个文章](https://zhuanlan.zhihu.com/p/609154637)

## FAQ

### 1. 能让AI生成文字的速度再快一点吗？

当然可以，把webui.py里面的os.environ["RWKV_CUDA_ON"] = '0'改成os.environ["RWKV_CUDA_ON"] = '1'，但是你的机器必须安装Visual C++生成工具，以及Nvidia的CUDA，CUDA比较好解决（可能还得装CUDNN，我没验证到底要不要，反正我是都装了），去官网下载就行了，建议安装11.7版本，这个Visual C++生成工具可以参考[这个链接](https://learn.microsoft.com/zh-cn/training/modules/rust-set-up-environment/3-install-build-tools)装好之后还需要配置一下环境变量，如下图：
![图片3](./pic/3.png)
我这里配置的值是C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64，你们根据实际情况进行配置，主要是找到cl.exe这个文件所在的文件夹，当然也要注意架构，不过一般来说，大家都是64位的系统了吧。这样就算是完成了，然后在运行脚本，你会发现文字的生成速度提高了很多。

### 2. 我在哪里可以下载的到模型呢？

当然是在[这里](https://huggingface.co/BlinkDL)，我比较推荐rwkv-4-raven里面的RWKV-4-Raven-7B-v6-ChnEng-2023xxxx-ctx2048之类的模型，当然其他的你们也可以试试，理论上说模型越大效果越好，但是因为14B的模型中文语料比较少，可能生成的效果不如7B的，所以暂时还是先用着7B吧，不过模型作者更新模型相当频繁，没事儿可以去看看，没准儿哪天就有了。

### 3. top_p、temperature、presence、frequency这四个参数有什么设置技巧吗？

老实说我也不明白，我代码里面预置的参数是参考[这篇文章](https://zhuanlan.zhihu.com/p/616353805)的设置，top_p越高，逻辑性越强，但是会降低多样性，temperature越低，逻辑性越强，但会降低多样性，这东西我们都需要摸索，也欢迎大家分享自己的经验。

### 4. 还有其他的一些经验吗？

这个模型的使用，老实说，我也在摸索中，不过对人物性格，背景故事进行详细的设定的确能够改善对话的体验，我看其他的一些模型的使用方法里面有提到说文字越丰富（初始的token量越大，当然也别过大了）AI说的话越丰富，通过调整示例对话里面的内容的确可以改变AI的说话方式，一定程度上也能改变性格，这个还挺重要的，一定要填写，但是格式一定要按照我预置的那个样子来，不然会出问题。