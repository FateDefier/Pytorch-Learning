import torch

# (1) Prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])   # (3,)  (1, 3)
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# (2) design model using class
class LinearModel(torch.nn.Module):
    """
    design model using class： 必须有 __init__ 和 forward
    """
    def __init__(self):  # 构造函数，初始化默认调用的函数
        # super 父类，super(名称, self).__init__()
        super(LinearModel, self).__init__()
        """
        class: torch.nn.Linear(in_feature, out_feature, bias=True)
        in_feature: size of each input sample (N, *, in_feature) * means any number of additional dimensions
        out_feature: size of each output sample (N, *, out_feature), 维度要与 y 相同
        bias: if set to False, the layer will not learn an additive bias.Default: True
        """
        self.linear = torch.nn.Linear(1, 1)  # 构造对象（实例化），包含权重 w 和偏置 b，linear 继承自Module
    """
    class Foobar:
        def __init__(self):
            pass
    
        def __call__(self, *args, **kwargs):
            print("Hello" + str(args[0]))

    foobar = Foobar()
    foobar(1, 2, 3)
    
    # *args 和 ** kwargs用处
    
    def func(*args, **kwargs):
        print(args)  # 不带 * 为 元组，带 * 号去掉括号
        print(*kwargs)  # 不带 * 为 字典，带 * 为 键值
    
    
    func(1, 2, 4, 3, x=3, y=4)
    """
    def forward(self, x):  # 必须写，overwrite（覆盖）父类函数 magic method __call__()
        y_hat = self.linear(x)  #
        return y_hat


model = LinearModel()  # model是callable 可调用的（可以写 model(x) 调用），实例化

# (3) construct loss and optimizer
"""
class: torch.nn.MSELoss(size_average=True, reduce=True)
size_average: 是否求均值，有可能最后一轮的 Mini-Batch 比前几轮少，这时候 True
reduce: 是否求和降维
注意：
现在只有一个参数reduction = 'sum'(对所有样本损失求和) / 'mean'(对所有样本损失求平均) / 'none'(保留每个样本的损失（张量）)
"""
criterion = torch.nn.MSELoss(reduction='sum')  # 会构建计算图
"""
class: torch.optim.SGD(params, lr=<object object>, momentum=0, weight_decay=0, nesterov=False)  
params: 哪些参数需要求梯度
lr: learning rate 学习率，支持不同批量使用不同的学习率
momentum: 冲量
weight_decay: 在优化的目标加上 (w^T · w)
nesterov: 
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 调整动量可以更快收敛
# LBFGS 是批量优化器，需要手动定义闭包函数
# for epoch in range(50):
#     def closure():
#         optimizer.zero_grad()
#         y_pred = model(x_data)
#         loss = criterion(y_pred, y_data)
#         loss.backward()
#         return loss
#     loss = closure()
#     print(epoch, loss)
#     optimizer.step(closure)

# (4) Training cycle
for epoch in range(50):  # 1.y_hat --> 2.loss --> 3.backward --> 4.update
    # 1. forward 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
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
print('y_pred=', y_test.data.item())  # y_test.data 返回 tensor
