import torch
from torch import nn
from torch.nn import *
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)

class Gan(nn.Module):
    def __init__(self):
        super(Gan,self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )     
    def forward(self,x):
        x = self.model1(x)
        return x 

loss = nn.CrossEntropyLoss()
gan = Gan()
optim = torch.optim.SGD(gan.parameters(),lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        output = gan(imgs)
        result_loss = loss(output,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
        # print("OK")
    print(running_loss)
# writer = SummaryWriter("./logs_seq")
# writer.add_graph(gan,input)
# writer.close()