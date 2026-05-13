import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True  # 需要计算梯度，默认的 Tensor 不计算


def forward(x):
    return x * w  # x 被转换成 Tensor，两个张量进行数乘，返回也为张量


def loss(xs, ys):
    y_pred = forward(xs)
    return (y_pred - ys) ** 2  # 返回张量


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l 为张量
        l.backward()  # 调用成员函数 —— backward 反馈，自动计算计算图上的所有梯度，并且计算图被释放
        print("\tgrad: ", x, y, w.grad.item())  # item() 张量取到标量，张量直接做加法运算会构建计算图
        w.data = w.data - 0.01 * w.grad.data  # w.grad也是张量，直接写实际是构建计算图，必须要取到data，修改数值

        w.grad.data.zero_()  # 梯度清零，如果不清 0 则每次的梯度会累加

    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())
