"""
Google Neural Network:
Inception模块：在同一个模块中，并行使用 1x1, 3x3, 5x5 卷积和池化层，让网络自己学习不同尺度特征的重要性
四条路径的 channel 可以不同，但是 width 和 height 要相同
1x1 convolution的作用：改变通道数，不改变 width, height，从而降低计算时间
例：
192x28x28 -->(5x5 Convolution) 32x28x28
Operations: 5^2 x 28^2 x192 x 32 = 120,422,400
192x28x28 -->(1x1 Convolution) 16x28x28 -->(5x5 Convolution) 32x28x28
Operations: 1^2 x 28^2 x192 x16 + 5^2 x 28^2 x 16 x 32 = 12,433,648
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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


class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)  # tensor的维度(B, C, W, H)依次为(第一维, 第二维, 第三维, 第四维),连接后为88


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)  # Inception输出的通道固定为88(24*3+16)

        self.inception1 = InceptionA(in_channels=10)
        self.inception2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(1408, 10)  # 1408 到Inception2得到

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.inception1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.inception2(x)
        x = x.view(in_size, -1)
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
[1,   300] loss: 0.126
[1,   600] loss: 0.028
[1,   900] loss: 0.020
Accuracy on test set: 97 % [9721/10000]
[2,   300] loss: 0.016
[2,   600] loss: 0.015
[2,   900] loss: 0.013
Accuracy on test set: 97 % [9771/10000]
[3,   300] loss: 0.011
[3,   600] loss: 0.011
[3,   900] loss: 0.010
Accuracy on test set: 97 % [9794/10000]
[4,   300] loss: 0.009
[4,   600] loss: 0.009
[4,   900] loss: 0.009
Accuracy on test set: 98 % [9850/10000]
[5,   300] loss: 0.008
[5,   600] loss: 0.008
[5,   900] loss: 0.008
Accuracy on test set: 98 % [9831/10000]
[6,   300] loss: 0.007
[6,   600] loss: 0.007
[6,   900] loss: 0.007
Accuracy on test set: 98 % [9876/10000]
[7,   300] loss: 0.006
[7,   600] loss: 0.006
[7,   900] loss: 0.006
Accuracy on test set: 98 % [9862/10000]
[8,   300] loss: 0.006
[8,   600] loss: 0.005
[8,   900] loss: 0.007
Accuracy on test set: 98 % [9888/10000]
[9,   300] loss: 0.005
[9,   600] loss: 0.005
[9,   900] loss: 0.006
Accuracy on test set: 98 % [9884/10000]
[10,   300] loss: 0.005
[10,   600] loss: 0.004
[10,   900] loss: 0.006
Accuracy on test set: 98 % [9881/10000]

Process finished with exit code 0
"""