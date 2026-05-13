"""
梯度消失：一连串的小于 1 的值相乘，最终的梯度接近于 0，更新时权重几乎无变化
解决方法：逐渐加隐藏层，每次的梯度固定住（太麻烦了）
Residual Net:
x -->(Conv/FC) relu -->(Conv/FC) relu --> H(x)
x -->(Conv/FC) -->(Conv/FC) --> F(x) --> relu(F(x) + x)  导数为 F(x)' + 1 避免梯度消失
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 使用激活函数 Relu
import torch.optim as optim

batch_size = 64  # 要使用 dataset and dataLoader

# 1.Prepare dataset
transform = transforms.Compose([  # 将多个图像变换（transform）操作串联 / 组合成一个有序的 “变换管道”（也可称为变换序列）
    # 把图像转换成图像张量(channel(通道) x width(宽度) x height(高度)) 28 x 28(pixel属于{0, ...255}) --> 1 x 28 x 28(pixel属于[0,1])
    transforms.ToTensor(),  # 实际还要加上批次，故为四阶张量 (N, c, w, h)即(N, 1, 28, 28)
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, transform=transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST('../data', train=False, transform=transform, download=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + x)


# 2.Design Model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.residual_block1 = ResidualBlock(16)
        self.residual_block2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        # Flatten data form (n, 1, 28, 28) to (n, 784)
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.residual_block1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.residual_block2(x)
        x = x.view(in_size, -1)  # flatten
        x = self.fc(x)
        return x


model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # cuda:0 可以选取不同的显卡
model.to(device)  # 把所有的模块转化为cuda tensor，迁移到GPU

# 3.Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 4.Train cycle and Test
def train(epoch):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 放到同一块显卡上
        # 梯度清零
        optimizer.zero_grad()
        # forward
        output = model(data)
        loss = criterion(output, target)
        # backward
        loss.backward()
        # update
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():  # test 过程不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, dim=1)  # 沿着第一个维度（列）找下标，返回最大值和对应的行下标
            total += target.size(0)
            correct += (predicted == target).sum().item()  # 张量间的比较训练
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test(epoch)

"""
输出结果：
[1,   300] loss: 0.074
[1,   600] loss: 0.022
[1,   900] loss: 0.016
Accuracy on test set: 97 % [9705/10000]
[2,   300] loss: 0.013
[2,   600] loss: 0.011
[2,   900] loss: 0.010
Accuracy on test set: 98 % [9820/10000]
[3,   300] loss: 0.010
[3,   600] loss: 0.008
[3,   900] loss: 0.008
Accuracy on test set: 98 % [9832/10000]
[4,   300] loss: 0.007
[4,   600] loss: 0.007
[4,   900] loss: 0.007
Accuracy on test set: 98 % [9880/10000]
[5,   300] loss: 0.006
[5,   600] loss: 0.005
[5,   900] loss: 0.006
Accuracy on test set: 98 % [9889/10000]
[6,   300] loss: 0.005
[6,   600] loss: 0.005
[6,   900] loss: 0.005
Accuracy on test set: 98 % [9884/10000]
[7,   300] loss: 0.004
[7,   600] loss: 0.005
[7,   900] loss: 0.004
Accuracy on test set: 98 % [9890/10000]
[8,   300] loss: 0.004
[8,   600] loss: 0.004
[8,   900] loss: 0.005
Accuracy on test set: 99 % [9917/10000]
[9,   300] loss: 0.003
[9,   600] loss: 0.004
[9,   900] loss: 0.004
Accuracy on test set: 98 % [9888/10000]
[10,   300] loss: 0.003
[10,   600] loss: 0.003
[10,   900] loss: 0.004
Accuracy on test set: 99 % [9905/10000]

Process finished with exit code 0
"""