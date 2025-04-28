import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

# 方式1》保存方式1，加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2》保存方式2，加载模型
# 要恢复网络模型结构
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# 通过字典形式保存的 model 参数
# model = torch.load("vgg16_method2.pth")
print(vgg16)
