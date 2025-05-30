# 完整的项目流程
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

# 搭建神经网络
class gan(nn.Module):
    def __init__(self):
        super(gan, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1,padding=2),   
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)            
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == "__main__":
    gan = gan()
    input = torch.ones((64, 3, 32, 32))
    output = gan(input)
    print(output.shape)