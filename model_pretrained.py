import torchvision
import torch
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d, Flatten, Sequential
from torch.utils.data import DataLoader

# train_data = torchvision.datasets.ImageNet("./data_image_net", split="train", download=True, transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False,progress=True)
vgg16_true = torchvision.models.vgg16(pretrained=True,progress=True)
print("OK")
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

# 根据现有的网络改善其结构，在现有的网络上添加新的结构，如：
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))

print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)