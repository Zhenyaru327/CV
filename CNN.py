"""
卷积神经网络
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import argmax

#定义一些参数
batch_size = 64
EPOCH = 5
learning_rate = 0.01

#数据预处理
#compose可以把一系列的操作整合起来

from torchvision import transforms
#第一个操作是将期转化为Tensor
data_tf = transforms.Compose(
    [transforms.ToTensor(),#把图像转化为张量
    transforms.Normalize([0.1307],[0.3081])]#均值和标准差，切换到0-1分布用来训练
)

#下载数据集并且进行整合
from torchvision import datasets
from torch.utils.data import DataLoader
train_dataset = datasets.MNIST(r'.\data',train=True,download=True,transform=data_tf)
test_dataset = datasets.MNIST(r'.\data',train=False,download=True,transform=data_tf)

#继续使用Dataloader读取数据
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

print(len(train_loader))
print(len(test_loader))

#展示图片
examples = enumerate(test_loader)  # img&label
#enumerate将图片和下标整合在一起,(img,labels)
batch_idx, (imgs, labels) = next(examples)  # 读取数据,batch_idx从0开始

print(labels)  # 读取标签数据
# print(labels.size(0))
print(labels.size())  # torch.Size([64])，因为batch_size为64


fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(imgs[i][0], cmap='gray', interpolation='none')
    plt.title("Truth: {}".format(labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# #构建卷积神经网络的模型
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,25,kernel_size=3),#1表示的是输入的通道数，25表示输出的通道数，3是卷积核的大小
            torch.nn.BatchNorm2d(25),#对数据进行归一化处理，使得在进行Relu激活之前不至于数据太大而使网络性能不稳定
            torch.nn.ReLU(inplace=True)#inplace表示是否原地执行，选择进行原地覆盖操作
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(25,50,kernel_size=3),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(inplace=True)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        #全连接
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(50*5*5,1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024,128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x

model = CNN()

if torch.cuda.is_available():
    model = model.cuda()

#定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

idx = 0
costs = []
for epoch in range(EPOCH):

    for data in train_loader:
        optimizer.zero_grad()
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        out = model(img)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()

        idx += 1
        costs.append(loss.data.item())
        if idx % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch+1, loss.data.item()))

plt.figure()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(range(idx),costs)
plt.show()

#模型评估
model.eval()
eval_loss = 0
eval_acc = 0
epochs = 0
e_loss = []
e_acc = []
for data in test_loader:
    img, label = data
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    out = model(img)
    loss = criterion(out, label)
    epochs += 1
    e_loss.append(loss.data.item())
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)#torch.max()函数返回两个值，一个是具体的值，也就是预测概率，另一个是值对应的索引，即预测类别
    num_correct = (pred == label).sum()
    e_acc.append(num_correct.item()/64)
    eval_acc += num_correct.item()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_dataset)),
    eval_acc / (len(test_dataset))
))

# print(e_acc)
plt.figure()
plt.plot(range(epochs),e_acc)
plt.xlabel("epochs")
plt.ylabel("acc")
plt.show()

fig = plt.figure(figsize=(20, 20))
for i in range(20):
    # 随机抽取20个样本
    index = random.randint(0, len(test_loader))
    data = test_loader.dataset[index]
    if torch.cuda.is_available():
        img = data[0].cuda()
    else:
        img = data[0]
    img = torch.reshape(img, (1, 1, 28, 28))
    with torch.no_grad():
        output = model(img)
    # plot the image and the predicted number
    fig.add_subplot(4, 5, i + 1)
    #argmax用于对一个numpy数据取最大值的一个下标，若有多个最大值，则取第一个最大值的下标
    plt.title(argmax(output.data.cpu().numpy()))
    plt.imshow(data[0].numpy().squeeze(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()