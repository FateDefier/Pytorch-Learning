# PyTorch 深度学习实践 — 学习笔记

基于 B 站 [刘二大人《PyTorch深度学习实践》](https://www.bilibili.com/video/BV1Y7411d7Ys) 课程的逐课代码实现，从 NumPy 手写线性回归到 GoogLeNet/GRU 等现代架构，适合零基础入门深度学习。

---

## 目录

- [项目结构](#项目结构)
- [学习路线（按编号顺序）](#学习路线按编号顺序)
- [模型架构速览](#模型架构速览)
- [配套文档](#配套文档)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [学习建议](#学习建议)
- [致谢](#致谢)

---

## 项目结构

```
Pytorch-Learning/
├── 1. Linear_model(Numpy).py              # NumPy 手写线性回归
├── 2. Linear_model(Torch).py             # PyTorch 版线性回归
├── 3. Stochastic_Gradient_Decent.py      # 随机梯度下降（手写）
├── 4. Batch_Gradient_Decent.py           # 批量梯度下降（手写）
├── 5. Back_Propagation.py                # 反向传播与 autograd
├── 6. BP_Practice.py                     # 反向传播实战（二次模型）
├── 7. Logistic_Regression(Classification).py  # 二分类 + Sigmoid
├── 8. Multiple_Dimentional_Logistic_Regression.py  # 多维逻辑回归 + 激活函数对比
├── 9. Softmax_Classifier(Multi-class).py # 多分类 + Softmax + 全连接网络
├── 10. Dataset_and_DataLoader.py         # 自定义数据集与 DataLoader
├── 11. Convolutional_Neural_Network.py   # 卷积神经网络（CNN）
├── 12. Residual_Net.py                   # 残差网络（ResNet）
├── 13. GoogleNet(Inception).py           # GoogLeNet + Inception 模块
├── 14. Recurrent_Neural_Network.py       # 循环神经网络（RNN）
├── 15. Gated_Recurrent_Unit.py           # 双向 GRU + 序列分类
├── Activation function.md                # 🔥 10+ 种激活函数完全指南
├── Different Optimizer.md                # 🔥 8 种优化器对比与选型指南
└── README.md                             # 本文件
```

---

## 学习路线（按编号顺序）

每个文件对应课程的一讲，**建议严格按编号顺序学习**——后一个文件会用到前一个文件的概念。

### 第一阶段：基础工具（文件 1-2）

| # | 文件 | 学什么 | 亮点 |
|---|------|--------|------|
| 1 | `1. Linear_model(Numpy).py` | 线性回归纯 NumPy 实现、MSE 损失、3D 可视化 | 理解"训练循环"的底层机制 |
| 2 | `2. Linear_model(Torch).py` | PyTorch 版线性回归、`nn.Module`、`MSELoss`、`optim.SGD` | 从手写过渡到框架 |

### 第二阶段：梯度下降与反向传播（文件 3-6）

| # | 文件 | 学什么 | 亮点 |
|---|------|--------|------|
| 3 | `3. Stochastic_Gradient_Decent.py` | 随机梯度下降手写实现，**参数更新公式** `θ ← θ − lr·∇J` | 每个样本更新一次权重 |
| 4 | `4. Batch_Gradient_Decent.py` | 批量梯度下降手写实现 | 全量数据计算梯度，对比 SGD |
| 5 | `5. Back_Propagation.py` | PyTorch 自动求导（`autograd`）、计算图 | 告别手算梯度 |
| 6 | `6. BP_Practice.py` | 反向传播实战：`y = w₁x² + w₂x + b` | 多参数 + autograd 联合训练 |

### 第三阶段：分类问题（文件 7-9）

| # | 文件 | 学什么 | 数据集 |
|---|------|--------|--------|
| 7 | `7. Logistic_Regression(Classification).py` | 二分类、Sigmoid 激活、BCELoss、决策边界可视化 | 合成数据（3 个样本） |
| 8 | `8. Multiple_Dimentional_Logistic_Regression.py` | 多维逻辑回归、**7 种激活函数对比实验**（Sigmoid/ReLU/Tanh/LeakyReLU/ELU/Swish/Softplus） | 糖尿病数据集 (8→1) |
| 9 | `9. Softmax_Classifier(Multi-class).py` | 多分类、Softmax、CrossEntropyLoss、**5 层全连接网络**（784→512→256→128→64→10） | MNIST（手写数字） |

### 第四阶段：数据处理与工程化（文件 10）

| # | 文件 | 学什么 | 亮点 |
|---|------|--------|------|
| 10 | `10. Dataset_and_DataLoader.py` | 自定义 `Dataset`、`DataLoader` 小批量训练、Mini-Batch、Epoch/Iteration 概念 | 从全量数据 → mini-batch 训练 |

### 第五阶段：现代神经网络架构（文件 11-15）

| # | 文件 | 学什么 | 数据集 | 准确率 |
|---|------|--------|--------|:------:|
| 11 | `11. Convolutional_Neural_Network.py` | 卷积层、池化层、多通道卷积、GPU 迁移 | MNIST | **~98%** |
| 12 | `12. Residual_Net.py` | 残差块（ResidualBlock）、跳跃连接、**解决梯度消失** | MNIST | **~99%** |
| 13 | `13. GoogleNet(Inception).py` | Inception 模块、1×1 卷积降维、多尺度特征融合 | MNIST | **~98%** |
| 14 | `14. Recurrent_Neural_Network.py` | RNNCell 与 RNN、Embedding 层、序列预测、独热编码 | "hello" → "ohlol" | — |
| 15 | `15. Gated_Recurrent_Unit.py` | 双向 GRU、PackedSequence 变长序列处理、人名国籍分类 | 18 国人名 | — |

---

## 模型架构速览

### 全连接网络（文件 9）

```
Input(784) → Linear(512) → ReLU → Linear(256) → ReLU
           → Linear(128) → ReLU → Linear(64) → ReLU
           → Linear(10) → CrossEntropyLoss
```
5 层全连接，MNIST 测试准确率 **~97%**。

### 卷积神经网络（CNN，文件 11）

```
Conv(1→10, 5×5) → ReLU → MaxPool(2×2)
Conv(10→20, 5×5) → ReLU → MaxPool(2×2)
Flatten → Linear(320→128) → ReLU → Linear(128→10)
```
两层卷积 + 池化 + 全连接，MNIST 测试准确率 **~98%**。

### 残差网络（ResNet，文件 12）

```
Conv(1→16, 5×5) → MaxPool(2×2)
ResBlock(16→16) → ResBlock(16→32)   ← 跳跃连接：F(x)+x
AveragePool → Linear(32→10)
```
通过跳跃连接（F(x) + x）让梯度直接流过，导数 = F'(x) + 1，**有效避免梯度消失**。MNIST 测试准确率 **~99%**。

### GoogLeNet / Inception（文件 13）

```
Conv(1→16, 5×5) → MaxPool(2×2)
InceptionModule(16→24) → InceptionModule(24→64)
AveragePool → Linear(64→10)
```
Inception 模块并行使用 1×1、3×3、5×5 卷积和池化，让网络自己学习不同尺度特征的重要性。其中 1×1 卷积用于**降维**（减少计算量 ~10 倍）。MNIST 测试准确率 **~98%**。

### 循环神经网络（RNN，文件 14）

```
Embedding(vocab_size→4) → RNN(4→4, num_layers=1)
→ Linear(4→vocab_size) → CrossEntropyLoss
```
输入 "hello"，学习预测下一个字符。展示了 RNNCell 逐时间步手动循环和 RNN 自动循环两种用法。

### 双向 GRU（文件 15）

```
Embedding(vocab_size→100) → BiGRU(100→100, num_layers=2)
→ Linear(200→num_classes)
```
双向 GRU（前向 + 反向），配合 PackedSequence 处理变长名字序列，分类 18 个国家的人名。

---

## 配套文档

| 文档 | 内容 |
|------|------|
| [`Activation function.md`](Activation%20function.md) | 10+ 种激活函数（Sigmoid/Tanh/ReLU/LeakyReLU/PReLU/ELU/SELU/Swish/GELU/Mish/Softmax）的**公式、PyTorch 代码、优缺点、选型速查表、常见问题 FAQ** |
| [`Different Optimizer.md`](Different%20Optimizer.md) | 8 种优化器（SGD/Adagrad/RMSprop/Adam/Adamax/ASGD/Rprop/LBFGS）的**原理、公式推导、PyTorch 调用、性能对比实验、选型决策流程** |

> 📖 两篇文档均为**独立阅读材料**，不依赖代码文件。初学者建议在学习到相应模块时配合阅读。

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| **Python** | ≥ 3.8 | 运行环境 |
| **PyTorch** | ≥ 1.7 | 核心深度学习框架 |
| **torchvision** | ≥ 0.8 | 数据集下载（MNIST）与图像变换 |
| **NumPy** | ≥ 1.19 | 数值计算（部分文件手写实现用） |
| **Pandas** | ≥ 1.2 | CSV 数据加载（糖尿病数据集） |
| **Matplotlib** | ≥ 3.3 | 损失曲线、决策边界、3D 可视化 |

安装命令：

```bash
pip install torch torchvision numpy pandas matplotlib
```

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/xiu-yuan-siu/Pytorch-Learning.git
cd Pytorch-Learning

# 2. 安装依赖
pip install torch torchvision numpy pandas matplotlib

# 3. 按编号顺序运行
python "1. Linear_model(Numpy).py"
python "2. Linear_model(Torch).py"
# ...
```

### 数据文件说明

部分文件需要外部数据集，代码中已配置自动下载或从本地加载：

| 文件 | 所需数据 | 获取方式 |
|------|----------|----------|
| 文件 1-7 | `y=2x` 合成数据 | ✅ 代码内生成，无需额外下载 |
| 文件 8、10 | `diabetes.csv` | 放在 `../data/` 或修改路径为项目根目录 |
| 文件 9、11-13 | MNIST | ✅ `torchvision.datasets.MNIST` 自动下载到 `./dataset/mnist/` |
| 文件 15 | `names_train.csv.gz` / `names_test.csv.gz` | 放在 `../data/` 目录 |

---

## 学习建议

1. **按顺序学**：每个文件都建立在前一个的基础上。跳过中间文件直接看 ResNet 会缺失关键概念。
2. **动手改参数**：学习率、batch_size、隐藏层神经元数——改一改看看效果，比只看代码有效 10 倍。
3. **配合文档阅读**：学到激活函数时看 [`Activation function.md`](Activation%20function.md)，学到优化器时看 [`Different Optimizer.md`](Different%20Optimizer.md)。
4. **先看懂训练循环**：文件 1-5 的核心都在这个模式上：
   ```python
   for epoch in range(epochs):
       y_pred = model(x)              # 1) 前向传播
       loss = loss_fn(y_pred, y)      # 2) 计算损失
       optimizer.zero_grad()          # 3) 梯度清零
       loss.backward()                # 4) 反向传播
       optimizer.step()               # 5) 更新参数
   ```

---

## 致谢

感谢 [刘二大人](https://space.bilibili.com/397680827) 的《PyTorch深度学习实践》课程——用最简洁的代码讲最深的概念，是中文深度学习社区不可多得的入门资源。
