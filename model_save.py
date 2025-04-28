import torchvision
import torch
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d, Flatten, Sequential

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1，模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐） 麻烦一些但是更标准
torch.save(vgg16.state_dict(), "vgg16_method2.pth")