import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def model(x, w, b):
    return x * w + b  # Affine model 仿射模型，也叫一个线性单元(linear unit)


def loss(x, y, w, b):
    y_pred = model(x, w, b)
    return (y_pred - y) ** 2


w_arr = np.arange(0.0, 4.1, 0.1)
b_arr = np.arange(-2.0, 2.0, 0.1)
Z = np.zeros((len(b_arr), len(w_arr)))  # 注意这里的顺序，b在前，w在后，与 meshgrid 生成的网格对照

for idx_w, w in enumerate(w_arr):
    for idx_b, b in enumerate(b_arr):
        l_sum = 0
        for x, y in zip(x_data, y_data):
            y_hat = model(x, w, b)
            loss_val = loss(x, y, w, b)
            l_sum += loss_val
            print('\t', x, y, round(y_hat, 2), round(loss_val, 2))
        Z[idx_b, idx_w] = l_sum / len(x_data)  # MSE，这里也要 b 在前，w 在后

W, B = np.meshgrid(w_arr, b_arr)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, Z, rstride=1, cstride=1, cmap='viridis', alpha=0.8)

# 添加颜色条，直观显示损失值
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('w(weight)')
ax.set_ylabel('b(bias)')
ax.set_zlabel('MSE')

plt.show()
