# import torchvision
#
# """
# Sigmoid activation function: 激活函数
# Logistic function
#
# """
# """
# MNIST 数据集:
# Training Set: 60000 examples
# Test Set: 10000 examples
# class = 10(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# """
#
# # 训练集(60000 example)，从官网下--True
# train_set = torchvision.datasets.MNIST(root='./dataset/mnist', train=True, download=True)
# # 测试集(10000 example)
# test_set = torchvision.datasets.MNIST(root='./dataset/mnist', train=False, download=True)
# """
# CIFAR10 数据集:
# training set: 50000 examples
# test set: 10000 examples
# class = 10(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
# """
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# (1) Prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])  # 注意这是分类问题


# (2) design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):  # 构造函数，初始化默认调用的函数
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 构造对象（实例化），包含权重 w 和偏置 b，linear 继承自Module

    def forward(self, x):  # 必须写，overwrite（覆盖）父类函数 magic method __call__()
        y_hat = F.sigmoid(self.linear(x))
        return y_hat


model = LogisticRegressionModel()  # model是callable 可调用的（可以写 model(x) 调用），实例化

# (3) construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# (4) Training cycle
# 1.y_hat --> 2.loss --> 3.backward --> 4.update
for epoch in range(1000):
    # 1. forward 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)
    optimizer.zero_grad()  # 梯度归零
    # 2. backward
    loss.backward()
    # 3. update
    optimizer.step()

# Output weight and bias
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

# Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=', y_test.data)  # y_test.data 返回 tensor

# (5) 可视化
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))  # 转换成 200 行 1 列的矩阵，view 相当于 numpy 的 reshape
y_t = model(x_t)
y = y_t.data.numpy()  # 拿到数据，然后转换成 numpy 的数组
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
