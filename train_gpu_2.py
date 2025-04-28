# 完整的项目流程
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from model import *
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
import time

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

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
gan = gan()
gan = gan.to(device)
    
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
# learning_rate = 1e-2  
# 1e-2 = 1 x (10)^(-2) = 1 /100 = 0.01
optimizer = torch.optim.SGD(gan.parameters(),lr=0.007)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加tensorboard
writer = SummaryWriter("./logs_train")
# 记录时间
strat_time = time.time()

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    gan.train()
    for data in train_dataloader:
        # 读取数据 - > 数据加载器 - > 损失函数
        imgs, targets = data
        imgs = imgs.to(device) # 数据使用gpu
        targets = targets.to(device) # 数据使用gpu
        outputs = gan(imgs)
        loss = loss_fn(outputs, targets)
        
        # 优化器优化模型
        # 梯度清零 - > 反向传播 - > 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录训练次数
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time() # 记录结束时间
            print(end_time - strat_time) # 计算训练时间
            print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # 测试步骤开始
    gan.eval() # 不启用 Batch Normalization 和 Dropout
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device) # 数据使用gpu
            targets = targets.to(device) # 数据使用gpu
            outputs = gan(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() # argmax(1) 返回每一行(横向)的最大值的索引
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    print("整体测试集上的Loss:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
        
    torch.save(gan, "gan_{}.pth".format(i))
    # print("模型已保存")
    
writer.close()
        