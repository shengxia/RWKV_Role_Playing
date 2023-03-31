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

如果运行不起来，把webui.py里面的os.environ["RWKV_CUDA_ON"] = '1'改成os.environ["RWKV_CUDA_ON"] = '0',cuda kernel编译问题不同机器报的错都不太一样，解决方案也不太一样，我也没细研究过，就不在这里多说了。