# 深度学习Pytorch（三）

## 一、Transforms的使用

![](https://xuyuya.oss-cn-guangzhou.aliyuncs.com/img_for_typora/20230310110742.png)

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# Python的用法-》tensor数据类型
# 通过transforms.ToTensor去看两个问题

# 2、为什么我们需要Tensor数据类型

# 绝对路径 https://xuyuya.oss-cn-guangzhou.aliyuncs.com/img_for_typora/20230310104902.png
# 相对路径 dataset/train/ants/0013035.jpg
img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

# 1、transforms该如何使用(Python)
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
writer.add_image("Tensor_img",tensor_img)

writer.close()
```

run后，打开终端，输入 `	 ` 即可查看结果

常见Transforms的使用：

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer=SummaryWriter("logs")
img=Image.open("dataset/train/ants/0013035.jpg")
print(img)

# ToTensor
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize归一化
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([1,3,5],[3,2,1])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Nomalize",img_norm)

#Resize
print(img.size)
trans_resize=transforms.Resize(768)
#img PIL -> resize -> img_resize PIL
img_resize=trans_resize(img)
#img_resize PIL -> totensor -> img_resize tensor
writer.add_image("Resize",img_resize,0)
print(img_resize)

#Compose - resize - 2
trans_size_2 = transforms.Resize(512)
#PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_size_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

#RandomCrop随机裁剪
trans_random = transforms.RandomCrop(512)
trans_compose_2 = trans_compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)  
writer.close()
```



## 二、torchvison中数据集的使用

官方文档：https://pytorch.org/vision/stable/datasets.html#

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("p10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()
```



## 三、DataLoader的使用

```python
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#准备的测试数据集
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor())
# batch_size一次抓取多少张，shuffle是否在两轮抓取中打乱顺序，num_workers用多少进程取数据，drop_last是否丢弃最后一次抓取不足的数据
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

#测试数据集中第一张图片及target
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()

```

**更多内容请关注我的博客：[bgemini.com](https://bgemini.com/)**
