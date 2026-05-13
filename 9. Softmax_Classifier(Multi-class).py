"""
多分类问题输出要求：
1.每种输出的概率 >= 0
2.各种输出的概率之和 = 1（各种输出是竞争性的，输出的是一个分布）

Softmax Layer 如何实现上述要求：
Softmax Function：$p(y=i)=\frac{e^{Z_i}}{\sum\limits_{j=0}^{K-1}e^{Z_j}}$
代码实现：
import numpy as np
y = np.array([1, 0, 0])
z = np.array([0.2, 0.1, -0.1])
y_hat = np.exp(z) / np.exp(z).sum()
loss = (- y * np.log(y_hat)).sum()
print(loss)
即 每个概率作为常用指数函数的指数求得结果，然后各值占所有值之和的比例（使用指数函数因为 e^x > 0）

Loss Function：与对应的独热向量求交叉熵
Loss Function = - Y log(Y_hat)（Negative Log-Likelihood Loss 负对数似然损失）

Torch.nn.CrossEntropyLoss() 包含 Softmax + Loss
注意其对应的 y 要是 长整型的 Tensor(LongTensor)
import torch

criterion = torch.nn.CrossEntropyLoss()
Y = torch.LongTensor([2, 0, 1])

Y_hat1 = torch.Tensor([[0.1, 0.2, 0.9],
                       [1.1, 0.1, 0.2],
                       [0.2, 2.1, 0.1]])

Y_hat2 = torch.Tensor([[0.8, 0.2, 0.3],
                       [0.2, 0.3, 0.5],
                       [0.2, 0.2, 0.5]])

l1 = criterion(Y_hat1, Y)
l2 = criterion(Y_hat2, Y)
print("Batch Loss1 = ", l1.data, "\nBatch Loss2 = ", l2.data)
输出结果：
Batch Loss1 =  tensor(0.4966)
Batch Loss2 =  tensor(1.2389)
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
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不做激活，直接给 Softmax


model = Net()

# 3.Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 4.Train cycle and Test
def train(epoch):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
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
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():  # test 过程不需要计算梯度
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, dim=1)  # 沿着第一个维度（列）找下标，返回最大值和对应的行下标
            total += target.size(0)
            correct += (predicted == target).sum().item()  # 张量间的比较训练
    print('Accuracy on test set: %.2f %% [%d/%d]' % (round(100 * correct / total, 2), correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test(epoch)
# 全连接神经网络忽略了一些因素，最好先做特征提取（人工提取方法：傅里叶变换FFT，小波wavelet,深度学习Auto，如CNN）再
"""
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ../data\MNIST\raw\train-images-idx3-ubyte.gz
100%|██████████| 9.91M/9.91M [04:35<00:00, 35.9kB/s]
Extracting ../data\MNIST\raw\train-images-idx3-ubyte.gz to ../data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ../data\MNIST\raw\train-labels-idx1-ubyte.gz
100%|██████████| 28.9k/28.9k [00:00<00:00, 48.4kB/s]
Extracting ../data\MNIST\raw\train-labels-idx1-ubyte.gz to ../data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ../data\MNIST\raw\t10k-images-idx3-ubyte.gz
100%|██████████| 1.65M/1.65M [00:27<00:00, 59.7kB/s]
Extracting ../data\MNIST\raw\t10k-images-idx3-ubyte.gz to ../data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ../data\MNIST\raw\t10k-labels-idx1-ubyte.gz
100%|██████████| 4.54k/4.54k [00:00<00:00, 1.81MB/s]
Extracting ../data\MNIST\raw\t10k-labels-idx1-ubyte.gz to ../data\MNIST\raw

[1,   300] loss: 2.234
[1,   600] loss: 1.002
[1,   900] loss: 0.435
Accuracy on test set: 90 %
[2,   300] loss: 0.321
[2,   600] loss: 0.280
[2,   900] loss: 0.246
Accuracy on test set: 93 %
[3,   300] loss: 0.198
[3,   600] loss: 0.183
[3,   900] loss: 0.163
Accuracy on test set: 95 %
[4,   300] loss: 0.135
[4,   600] loss: 0.132
[4,   900] loss: 0.127
Accuracy on test set: 96 %
[5,   300] loss: 0.100
[5,   600] loss: 0.103
[5,   900] loss: 0.100
Accuracy on test set: 96 %
[6,   300] loss: 0.082
[6,   600] loss: 0.082
[6,   900] loss: 0.075
Accuracy on test set: 97 %
[7,   300] loss: 0.067
[7,   600] loss: 0.062
[7,   900] loss: 0.061
Accuracy on test set: 97 %
[8,   300] loss: 0.051
[8,   600] loss: 0.051
[8,   900] loss: 0.053
Accuracy on test set: 97 %
[9,   300] loss: 0.045
[9,   600] loss: 0.042
[9,   900] loss: 0.039
Accuracy on test set: 97 %
[10,   300] loss: 0.029
[10,   600] loss: 0.038
[10,   900] loss: 0.036
Accuracy on test set: 97 %

Process finished with exit code 0
"""