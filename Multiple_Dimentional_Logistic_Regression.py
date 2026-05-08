# import torch
# import pandas as pd
#
# df = pd.read_csv('diabetes.csv')
#
# x_np = df.iloc[:, :-1].values
# y_np = df.iloc[:, [-1]].values  # -1 的 [] 不能省略否则和 y_hat 的维度不对应，不能计算损失
#
# x_data = torch.tensor(x_np, dtype=torch.float32)
# y_data = torch.tensor(y_np, dtype=torch.float32)
#
#
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear1 = torch.nn.Linear(8, 6)
#         self.linear2 = torch.nn.Linear(6, 4)
#         self.linear3 = torch.nn.Linear(4, 1)
#         self.sigmoid = torch.nn.Sigmoid()  # 来自 Module
#
#     def forward(self, x):
#         x = self.sigmoid(self.linear1(x))
#         x = self.sigmoid(self.linear2(x))
#         x = self.sigmoid(self.linear3(x))
#         return x
#
#
# model = Model()
#
# criterion = torch.nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#
# for epoch in range(100):  # 这里没有用 Mini-Batch 风格
#     # Forward
#     y_hat = model(x_data)
#     loss = criterion(y_hat, y_data)
#     print(epoch, loss.item())
#
#     # Backward
#     optimizer.zero_grad()
#     loss.backward()
#
#     # Update
#     optimizer.step()


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据加载（确保 diabetes.csv 在当前目录）
df = pd.read_csv('../data/diabetes.csv')
x_np = df.iloc[:, :-1].values
y_np = df.iloc[:, [-1]].values
x_data = torch.tensor(x_np, dtype=torch.float32)
y_data = torch.tensor(y_np, dtype=torch.float32)

# 通用训练函数（改为50 epochs）
def train_model(model, x_data, y_data, epochs=50, lr=0.2):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    for epoch in range(epochs):
        y_hat = model(x_data)
        loss = criterion(y_hat, y_data)
        loss_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_history

# 基础模型类
class BaseModel(torch.nn.Module):
    def __init__(self, hidden_act):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.hidden_act = hidden_act
        self.out_act = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.hidden_act(self.linear1(x))
        x = self.hidden_act(self.linear2(x))
        return self.out_act(self.linear3(x))

# 各种激活函数模型
models = {
    'Sigmoid': BaseModel(torch.nn.Sigmoid()),
    'ReLU': BaseModel(torch.nn.ReLU()),
    'Tanh': BaseModel(torch.nn.Tanh()),
    'LeakyReLU': BaseModel(torch.nn.LeakyReLU(0.1)),
    'ELU': BaseModel(torch.nn.ELU()),
    'Swish': BaseModel(torch.nn.SiLU()),
    'Softplus': BaseModel(torch.nn.Softplus())
}

# 训练并记录损失
loss_hist = {}
print("训练50轮结果对比：")
print(f"{'激活函数':<12}\t{'初始损失':<10}\t{'最终损失':<10}\t{'最小损失':<10}")
for name, model in models.items():
    hist = train_model(model, x_data, y_data, epochs=50, lr=0.2)
    loss_hist[name] = hist
    init_loss = hist[0]
    final_loss = hist[-1]
    min_loss = min(hist)
    print(f"{name:<12}\t{init_loss:.4f}\t\t{final_loss:.4f}\t\t{min_loss:.4f}")

# 可视化（50轮损失曲线）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
for i, (name, hist) in enumerate(loss_hist.items()):
    plt.plot(range(50), hist, label=name, color=colors[i], linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('BCELoss')
plt.title('不同激活函数前50轮损失曲线对比')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
