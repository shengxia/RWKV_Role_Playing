没有显卡？显存不足？内存不足？环境不会配置？现在都不用怕了，直接白嫖Google Colab吧！

## [点击这里快速到达](https://colab.research.google.com/drive/19cKRLE6WBVoVK1cHPNuc3KvmeQ9TO19G#scrollTo=t4daxu3L1Rbi)

嗯？你说你不会翻墙？不会翻墙你能上github？？？？？？

如果需要给角色加头像的话，可以把头像命名为"角色名.png"放在char目录下，至于头像去哪儿找嘛……用sd生成一个就好了。

## 一个基于RWKV的角色扮演玩具

![图片1](./pic/1.png)

就是这么一个玩意儿，连抄带编的弄出来了一个玩具，所以代码质量吗……请各位不要吐槽太多，但是也算是能玩吧。

### 安装方法：

先安装依赖
```
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 --upgrade

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
各种启动参数解释如下：

| 参数 | 解释 |
| --- | --- |
| --port | webui的端口 |
| --model | 要加载的模型路径 |
| --strategy | 模型加载的策略 |
| --listen | 加上这个参数就允许其他机器访问 |
| --cuda_on | 控制RWKV_CUDA_ON这个环境变量的，0-禁用，1-启用 |
| --jit_on | 控制RWKV_JIT_ON这个环境变量的，0-禁用，1-启用 |

模型的加载方式（--strategy）我默认使用的是"cuda fp16i8"，如果想使用其他的加载方式可以自行调整该参数，具体有哪些值可以参考[这个文章](https://zhuanlan.zhihu.com/p/609154637)或者这张图![图片](./pic/4.jpg)

## FAQ

### 1. 能让AI生成文字的速度再快一点吗？

当然可以，在启动命令中加入--cuda_on 1，例子：
```
python webui.py --listen --model model/fp16i8_RWKV-4-Pile-7B-EngChn-test5-20230326 --cuda_on 1
```
但是你的机器必须安装Visual C++生成工具，以及Nvidia的CUDA，CUDA比较好解决（可能还得装CUDNN，我没验证到底要不要，反正我是都装了），去官网下载就行了，建议安装11.7版本，这个Visual C++生成工具可以参考[这个链接](https://learn.microsoft.com/zh-cn/training/modules/rust-set-up-environment/3-install-build-tools)装好之后还需要配置一下环境变量，如下图：
![图片3](./pic/3.png)
我这里配置的值是C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64，你们根据实际情况进行配置，主要是找到cl.exe这个文件所在的文件夹，当然也要注意架构，不过一般来说，大家都是64位的系统了吧。这样就算是完成了，然后在运行脚本，你会发现文字的生成速度提高了很多。

### 2. 我在哪里可以下载的到模型呢？

当然是在[这里](https://huggingface.co/BlinkDL)，我比较推荐rwkv-4-raven里面的RWKV-4-Raven-7B-v6-ChnEng-2023xxxx-ctx2048之类的模型，当然其他的你们也可以试试，理论上说模型越大效果越好，但是因为14B的模型中文语料比较少，可能生成的效果不如7B的，所以暂时还是先用着7B吧，不过模型作者更新模型相当频繁，没事儿可以去看看，没准儿哪天就有了。

### 3. top_p、 top_k、temperature、presence、frequency这四个参数有什么设置技巧吗？

老实说我也不明白，我代码里面预置的参数是参考[这篇文章](https://zhuanlan.zhihu.com/p/616353805)的设置，top_p值越低，答案越准确而真实。更高的值鼓励更多样化的输出；temperature值越低，结果就越确定，因为总是选择概率最高的下一个词/token，拉高该值，其他可能的token的概率就会变大，随机性就越大，输出越多样化、越具创造性。
根据我这段时间的把玩，感觉top_p最好设置在0.5以上，temperature可以设置在1.5以上，presence和frequency设置成0.2或0.5随意，top_k可以设置为0，在对话刚开始，top_p可以设置的低一点，比如0.6或0.7，等对话一段时间，感觉有些无聊的时候，可以适当的调高top_p和temperature，在多次重说仍然无法得到想要的结果时，可以尝试调节top_k，我建议可以把top_k调到160~180左右，或者把已经在160~180之间的top_k重新调为0。

### 4. 模型会在输出回答后，又输出一大堆乱七八糟的内容。

这个问题我也在想办法改正，这的确是没有设置好停止词的问题，但目前我所知道的、有用的、有明显区分度的停止词还不多，对于这个我会继续优化代码，目前也只能在得到这样的回复之后让模型重说了，虽然降低top_p能够明显改善这个问题，但是降低top_p之后对话会变得非常无趣，得不偿失。

### 5. 为什么你的colab里要用fp16i8的7B模型？

没办法啊，免费的colab就给12G的内存，我用原版的7B模型，加载需要内存14G，colab它加载不进去啊！不过不用担心，性能、精度损失不大，再怎么说那也是7B模型，比3B的要好太多了。

### 6. 为什么不用流式输出？

在dev分支的某个版本中我尝试着使用了流式输出，虽然效果还不错，但是在对话久了之后，速度明显慢了，主要是传回的html代码大了，一句话十个字就得传10次，拖慢了速度，也有点耗费流量，感觉得不偿失（你说在本地用不怕耗流量？但是我喜欢用手机随时随地玩）当然，也可能是我的水平没到，写不出来增量传输。
我接下来想给这个项目开发一套API，到时候可以脱离gradio，使用html或者app来开发前端，这样更轻松，效果也更好，后续真有人想整个live2d之类的玩意儿对接这个模型，使用API也更方便一些。
