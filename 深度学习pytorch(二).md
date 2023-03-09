# 深度学习Pytorch（二）

前言：关于Pycharm终端显示PS而不显示虚拟环境名

解决办法：

1. 打开Pycharm的设置（File——>setting）,找到Tools，点击Terminal
2. 可以看到Shell path处给的路径是powershell.exe,所以终端才会一直显示PS
3. 将此处路径改为 `C:\Windows\system32\cmd.exe`,一般路径都是这个，改好之后点击OK即可
4. 关掉设置后重新打开终端，就可以看到自己的虚拟环境名了

## 一、Python文件、Python控制台、Jupyter的对比

1. 代码是以块为一个整体运行的话：

   Python文件：块是所有行的代码

   ​		优点：通用，传播方便，适用于大型项目

   ​		缺点：需要从头运行

   Python控制台：以任意行为块运行

   ​		优点：显示每个变量属性

   ​		缺点：不利于代码阅读及修改

   Jupyter：以任意行为块运行的

   ​		优点：利于代码阅读及修改

   ​		缺点：环境需要配置

   

## 二、Pytorch加载数据

Dataset类：

提供一种方式去获取数据及其label，它的功能是如何获取每一个数据及其label，并告诉我们总共有多少的数据

Dataloader类：

为后面的网络提供不同的数据形式

1. 在Pycharm中创建一个read_data的Python文件

   ```python
   from torch.utils.data import Dataset
   help(Dataset)  #或者直接使用Dataset??
   ```

   使用以上代码可以查看Dataset类的用法

2. 下面是读取数据的具体代码

   ```python
   from torch.utils.data import Dataset
   from PIL import Image
   import os
   class MyData(Dataset):
       def __init__(self,root_dir,label_dir):     #root_dirw为ants目录上层目录，label_dir为ants目录，此ants目录的目录名即为标签名
           self.root_dir=root_dir
           self.label_dir=label_dir
           self.path=os.path.join(self.root_dir,self.label_dir)   #拼接
           self.img_path=os.listdir(self.path)     #得到ants目录中所有图片的名字列表
   
       def __getitem__(self, idx):    #idx为编号
           img_name=self.img_path[idx]
           img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)   #拼接
           img=Image.open(img_item_path)
           label = self.label_dir
           return  img,label
       def __len__(self):
           return len(self.img_path)
   
   root_dir="dataset/train"
   ants_label_dir="ants"
   bees_label_dir="bees"
   ants_dataset=MyData(root_dir,ants_label_dir)   #创建蚂蚁数据集
   bees_dataset=MyData(root_dir,bees_label_dir)    #创建蜜蜂数据集
   train_dataset=ants_dataset+bees_dataset      #将蚂蚁数据集和蜜蜂数据集进行拼接
   img,label=train_dataset[0]      #相当于直接调用__getitem__函数，返回两个参数
   img.show()
   ```

   **注意：**这里采用的是以文件夹名来作为标签名的情况，如果标签名比较复杂，还有一种方式即采用两个文件夹，一个文件夹里放图片，另一个文件夹放对应名称的txt文件，每个txt文件里存有标签名。



## 三、TensorBoard的使用（add_scalar的使用）

**请先看前言！**

进入Pytorch环境后安装tensorboard：

```shell
pip install tensorboard
```

新建一个Python文件：
```python
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs")   #存储到logs文件夹下

# writer.add_image()
# y=x
for i in range(100):
    writer.add_scalar("y=x",i,i)    #第一个参数是标题，第二个参数是y轴，第三个参数是x轴
writer.close()
```

run以下以上代码后会生成一个logs文件夹，里面有我们的事件文件！

之后进入Pycharm终端，确保在Pytorch环境中，输入以下命令：

```shell
tensorboard --logdir=logs --port=6007   #logdir为事件文件所在文件夹名，port指定端口
```

会出现一个网址，进入即可得到咱们的scalar！

**注意：**每向wirter写入一个新的事件时，会保留上一次的事件，所以当我们需要变换函数的时候，一种方法是将logs文件夹中的事件文件全部删除，重新run！

## 四、TensorBoard的使用（add_image的使用）

**(后面学习用，这次不用)**进入Pytorch环境安装opencv：

```shell
pip install opencv-python
```

```python
from PIL import Image
image_path="dataset/train/ants/0013035.jpg"
img=Image.open(image_path)
print(type(img))
```

以上使用PIL获取图片，得到的图片格式是<class 'PIL.JpegImagePlugin.JpegImageFile'>

下面是add_image的三个参数，可以看到第二个参数只能使用三个类型，所以我们使用PIL的方式无法满足要求，但其中的numpy类型，可利用Opencv读取照片获得numpy型图片数据。

```python
Args:
            tag (str): Data identifier
            img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
            global_step (int): Global step value to record
```

此次不使用opencv获取图片，我们使用numpy.array()，对PIL图片进行转换

```
import numpy as np
img_array=np.array(img)
print(type(img_array))
```

结果是<class 'numpy.ndarray'>，符合要求

综上代码如下：

```python
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
writer=SummaryWriter("logs")   #存储到logs文件夹下

image_path="dataset/train/ants/69639610_95e0de17aa.jpg"
img_PIL=Image.open(image_path)    #得到PIL格式的图片
img_array=np.array(img_PIL)      #将PIL格式的图片转换为numpy格式的
print(type(img_array))
print(img_array.shape)
#从PIL到numpy，需要在add_image()中指定shape中每一个数字/维表示的含义
writer.add_image("train",img_array,1,dataformats='HWC')  #dataformats参数为高度H，宽度W,通道C，不清楚图片的类型就用图片.shape查看
#第二个参数为指定步数
# y=x
for i in range(100):
    writer.add_scalar("y=x",i,i)    #第一个参数是标题，第二个参数是y轴，第三个参数是x轴
writer.close()
```

run后进入Pycharm终端，确保在Pytorch环境中，输入以下命令：

```shell
tensorboard --logdir=logs --port=6007   #logdir为事件文件所在文件夹名，port指定端口
```

打开网页即可！



**更多内容请关注我的博客：[bgemini.com](https://bgemini.com/)**
