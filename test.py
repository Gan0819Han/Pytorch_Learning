from PIL import Image # 导入Image模块，用于打开图片并转换为RGB格式，防止报错：ValueError: cannot identify image file <_io.BytesIO object at 0x00000230F319C160>
import torchvision # 导入torchvision模块，用于定义transfor
import torch # 导入torch模块，用于定义transform
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

image_path = "D:\\VsCode\\Pytorch\\Source_Code\\src\\images\\dog1.png" # 图片路径
image = Image.open(image_path).convert('RGB') # 打开图片并转换为RGB格式
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()]) # 定义transform，将图片大小调整为32x32，并转换为张量格式

image = transform(image)
print(image.shape) # 输出图片张量的形状，应该为[3, 32, 32]，其中3表示通道数，32x32表示图片大小


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

model = torch.load("D:\\VsCode\\Pytorch\\Source_Code\\gan_6.pth") # 加载模型
print(model)

image = torch.reshape(image,(1,3,32,32))

model.eval() # 将模型设置为评估模式，即不进行梯度更新

with torch.no_grad(): # 关闭梯度计算，节省内存
    output = model(image)

print(output) # 输出模型的输出结果，即图片的类别
print("模型的输出为：{}".format(output.argmax(1))) # 输出模型的预测结果，即图片的类别

