"""
CNN(Convolutional Neural Network):
Feature Extraction(特征采样)：Convolution（卷积） + Subsampling（下采样）
Classification：flatten（展平） + Fully Connected（fc，全连接）

多通道卷积：input_channel = number_kernel(输入通道数 = 卷积核个数)，然后各结果相加（3 chanel --> 1 channel），这个过程称为卷积运算
完成通道数的改变 input_channel = n --> output_channel = m 的方法
若输入channel数为 n ，kernel 是 (n, 3, 3) 的 Tensor，这样的卷积核准备 m 个（即卷积核(m, n, kernel_size, kernel_size)），得到 m 个结果，然后 cat 起来，得到 (m, w', H')的图像
例如：
import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3  # 即 3 x 3
batch_size = 1

input = torch.randn(batch_size, in_channels, width, height)
conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
输出结果：
torch.Size([1, 5, 100, 100])
torch.Size([1, 10, 98, 98])
torch.Size([10, 5, 3, 3])

padding：根据输入和输出的 width 和 height，填充 n 圈（默认填充 0）
stride：跳格，有效降低大小(上下左右都跳格)
例如：
import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
input = torch.Tensor(input).view(1, 1, 5, 5)  # (B, C, W, H)

conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2, bias=False)

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)  # (O, I, W, H)
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)
输出结果(padding=1)：
tensor([[[[ 91., 168., 224., 215., 127.],
          [114., 211., 295., 262., 149.],
          [192., 259., 282., 214., 122.],
          [194., 251., 253., 169.,  86.],
          [ 96., 112., 110.,  68.,  31.]]]], grad_fn=<ConvolutionBackward0>)
输出结果(stride=2)：
tensor([[[[211., 262.],
          [251., 169.]]]], grad_fn=<ConvolutionBackward0>)

下采样(Subsampling): 常用如 Maxpooling（最大池化）不改变通道数
例如
import torch

input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6,]
input = torch.Tensor(input).view(1, 1, 4, 4)

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)  # 默认 stride=2

output = maxpooling_layer(input)
print(output)
输出结果：
tensor([[[[4., 8.],
          [9., 8.]]]])
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


# 2.Design Model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data form (n, 1, 28, 28) to (n, 784)
        batch_size1 = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))  # 卷据 --> 激活 --> 池化
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(batch_size1, -1)  # flatten
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

# 效果不好是因为最后的全连接层很少
"""
输出结果：
[1,   300] loss: 0.114
[1,   600] loss: 0.027
[1,   900] loss: 0.019
Accuracy on test set: 96 % [9658/10000]
[2,   300] loss: 0.016
[2,   600] loss: 0.014
[2,   900] loss: 0.013
Accuracy on test set: 97 % [9787/10000]
[3,   300] loss: 0.011
[3,   600] loss: 0.011
[3,   900] loss: 0.010
Accuracy on test set: 97 % [9791/10000]
[4,   300] loss: 0.009
[4,   600] loss: 0.010
[4,   900] loss: 0.008
Accuracy on test set: 98 % [9832/10000]
[5,   300] loss: 0.008
[5,   600] loss: 0.008
[5,   900] loss: 0.008
Accuracy on test set: 98 % [9830/10000]
[6,   300] loss: 0.007
[6,   600] loss: 0.007
[6,   900] loss: 0.007
Accuracy on test set: 98 % [9860/10000]
[7,   300] loss: 0.007
[7,   600] loss: 0.006
[7,   900] loss: 0.006
Accuracy on test set: 98 % [9867/10000]
[8,   300] loss: 0.006
[8,   600] loss: 0.005
[8,   900] loss: 0.006
Accuracy on test set: 98 % [9857/10000]
[9,   300] loss: 0.005
[9,   600] loss: 0.005
[9,   900] loss: 0.006
Accuracy on test set: 98 % [9858/10000]
[10,   300] loss: 0.005
[10,   600] loss: 0.005
[10,   900] loss: 0.004
Accuracy on test set: 98 % [9871/10000]

Process finished with exit code 0
"""
