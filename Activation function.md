# 激活函数完全指南 —— 从入门到选型

---

## 一、什么是激活函数？为什么需要它？

神经网络本质上是一层层矩阵乘法和加法的堆叠。如果没有激活函数，无论堆多少层，最终都等价于**一个线性变换**：

```
y = W₃(W₂(W₁x + b₁) + b₂) + b₃
  = (W₃W₂W₁)x + 常数   ← 还是线性！
```

激活函数的作用就是**引入非线性**，让网络能拟合任意复杂的函数。你可以把它理解为神经网络中的"开关"——决定某个神经元是否被"激活"。

### 核心要点（先记住这三点）

1. **没有激活函数 → 多层网络退化为单层线性模型**
2. **不同层可以用不同激活函数**（隐藏层用 ReLU，输出层用 Softmax/Sigmoid）
3. **好的激活函数 = 计算快 + 梯度不消失 + 输出稳定**

---

## 二、PyTorch 中使用激活函数

```python
import torch
import torch.nn as nn

# 方式一：作为模块（推荐，可被 model.parameters() 追踪）
layer = nn.ReLU()
output = layer(input_tensor)

# 方式二：作为函数（无状态，轻量）
import torch.nn.functional as F
output = F.relu(input_tensor)
```

---

## 三、激活函数全家福

### 3.1 经典三剑客：Sigmoid / Tanh / ReLU

| 激活函数 | 公式 | 输出范围 | 梯度范围 | 一句话总结 |
|:--------:|------|:--------:|:--------:|------------|
| **Sigmoid** | `σ(x) = 1 / (1 + e⁻ˣ)` | (0, 1) | (0, 0.25) | 把任意输入压到概率区间，最古老 |
| **Tanh** | `tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)` | (−1, 1) | (0, 1) | Sigmoid 的零均值升级版 |
| **ReLU** | `ReLU(x) = max(0, x)` | [0, +∞) | 0 或 1 | 简单粗暴，深度学习标配 |

#### 详解与 PyTorch 代码

**① Sigmoid** —— 适合**二分类输出层**

```python
# 二分类输出层用 Sigmoid + BCELoss
model = nn.Sequential(
    nn.Linear(10, 1),
    nn.Sigmoid()  # 输出 ∈ (0, 1)，可解释为概率
)
loss_fn = nn.BCELoss()
```

⚠️ **为什么隐藏层不用它？** 当输入 x 很大或很小时，Sigmoid 曲线几乎平了，梯度 → 0，参数更新停滞。这就是著名的"梯度消失"。如果 10 层都用 Sigmoid，最前面几层几乎学不到东西。

**② Tanh** —— Sigmoid 的改进版，**输出以零为中心**

```python
nn.Tanh()  # PyTorch 用法与 Sigmoid 完全相同
```

- 优点是零均值输出（Sigmoid 的均值是 0.5），梯度下降时收敛更稳定
- 缺点是一样的：x 远离 0 时梯度仍然消失

**③ ReLU（修正线性单元）** —— **隐藏层默认首选**

```python
nn.ReLU()  # 几乎所有 CNN/MLP 的隐藏层都用它
```

- ✅ x > 0 时梯度恒为 1（彻底解决梯度消失）
- ✅ 计算极快（就是一个 max 操作）
- ✅ 稀疏激活（x ≤ 0 直接输出 0），类似生物神经元
- ❌ **"死亡 ReLU"问题**：如果某个神经元输出恒 ≤ 0，梯度永远为 0，该神经元就"死了"，再也学不到东西

---

### 3.2 ReLU 变体：解决"死亡神经元"

| 激活函数 | 公式 | 改进点 | PyTorch |
|:--------:|------|--------|---------|
| **Leaky ReLU** | `max(0.01x, x)` | 负半轴保留微小梯度 | `nn.LeakyReLU(0.01)` |
| **PReLU** | `max(αx, x)`，α 可学习 | α 由训练自动调整 | `nn.PReLU()` |
| **ELU** | `x (x>0)`；`α(eˣ−1) (x≤0)` | 负半轴平滑 + 零均值 | `nn.ELU()` |
| **SELU** | 与 ELU 类似，`α≈1.673`，`λ≈1.051` | 自带"自归一化"特性 | `nn.SELU()` |

#### 详解

**① Leaky ReLU** —— 最简单的 ReLU 升级

```python
nn.LeakyReLU(negative_slope=0.01)  # α 通常取 0.01
```

当 x < 0 时输出 `0.01x` 而不是 0，梯度永远不为 0，避免"死亡"。**快速替换 ReLU 的首选**。

**② PReLU（Parametric ReLU）** —— 让模型自己学 α

```python
nn.PReLU()  # α 作为可学习参数
```

- 理论上比 Leaky ReLU 更灵活
- ⚠️ 小数据集上容易过拟合，适合 ImageNet 级别的大数据集

**③ ELU（指数线性单元）** —— 追求稳定性

```python
nn.ELU(alpha=1.0)
```

- 负半轴平滑指数衰减（而非直线），输出均值更接近 0
- 适合对训练稳定性要求高的场景（如 GAN）
- 代价：指数运算比 ReLU 的 max 稍慢

**④ SELU** —— 自动归一化的魔法

```python
nn.SELU()  # 需要配合特定的权重初始化（LeCun Normal）
```

- 在特定条件下，网络每层输出自动保持均值 0、方差 1
- 适合深层全连接网络，CNN 中少用

---

### 3.3 现代霸主：Swish / GELU / Mish

| 激活函数 | 公式 | 核心特点 | 代表用途 |
|:--------:|------|----------|----------|
| **Swish** | `x · σ(βx)` | 非单调、平滑 | EfficientNet |
| **GELU** | `x · Φ(x)` ≈ `x · σ(1.702x)` | 随机正则化思想 | BERT / GPT / ViT |
| **Mish** | `x · tanh(softplus(x))` | 比 ReLU 更平滑 | YOLOv4 |

#### 详解

**① Swish（自门控激活）** —— Google 出品

```python
# PyTorch 没有内置 Swish，可以自己定义
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)  # x 从"开关"变成了"门控"！

# 或用 SiLU（PyTorch 1.7+ 已内置，就是 β=1 的 Swish）
nn.SiLU()
```

- 数学上就是 `x·σ(x)`，像 ReLU 的平滑版本
- "非单调"意味着曲线有轻微凹陷，这种特性在深层网络中能保留更多信息
- EfficientNet 用它刷榜 ImageNet

**② GELU（高斯误差线性单元）** —— Transformer 标配

```python
nn.GELU()  # PyTorch 内置
```

- BERT、GPT、ViT 等几乎所有 Transformer 模型的默认激活函数
- 背后思想：加入"随机正则化"，输入越大，被激活的概率越高（而非 ReLU 的确定性开关）
- 近似公式 `x · σ(1.702x)`，计算量比 Swish 略高但效果更好

**③ Mish** —— 计算机视觉的新宠

```python
# 自定义实现
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))  # softplus = ln(1+eˣ)
```

- YOLOv4 中使用后精度提升明显
- 比 Swish/GELU 更平滑，代价是计算量更大

---

### 3.4 输出层专用：Softmax

| 激活函数 | 公式 | 输出特点 | 适用场景 |
|:--------:|------|----------|----------|
| **Softmax** | `pᵢ = e^zᵢ / Σe^zⱼ` | 输出 ∈ (0,1)，所有输出之和 = 1 | 多分类最后一层 |
| **Sigmoid** | `1/(1+e⁻ˣ)` | 单个值 ∈ (0,1) | 二分类最后一层 |

```python
# 多分类
model = nn.Sequential(
    nn.Linear(784, 10),
    nn.Softmax(dim=1)  # 10 个类别的概率分布
)
# 但通常直接用 CrossEntropyLoss（它内置了 Softmax）
loss_fn = nn.CrossEntropyLoss()  # 输入 logits，内部自动 Softmax

# 二分类
model = nn.Sequential(
    nn.Linear(784, 1),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
```

⚠️ `nn.CrossEntropyLoss()` 内部已经包含 Softmax，**不要再在模型里加 Softmax 层**，否则会 Softmax 两次导致梯度异常。

---

## 四、选型速查表

| 场景 | 推荐激活函数 | 原因 |
|------|:----------:|------|
| MLP / CNN **隐藏层**（通用） | **ReLU** 或 **Leaky ReLU** | 快、不消失、够用 |
| ReLU 出现大量死亡神经元 | **Leaky ReLU** / **ELU** | 保留负半轴梯度 |
| Transformer 类模型 | **GELU** | 行业标准，BERT/GPT 都用它 |
| 计算机视觉（追求极致精度） | **Mish** 或 **Swish** | 平滑非单调，深层网络更稳 |
| **二分类**输出层 | **Sigmoid** + BCELoss | 输出天然是概率 |
| **多分类**输出层 | 直接用 `CrossEntropyLoss` | 内部含 Softmax，别重复加 |
| 回归问题输出层 | **不用激活函数**（线性） | 输出需要覆盖任意实数 |

### 决策流程

```
你的任务是？
├── 回归（预测连续值）
│   └── 输出层：不用激活函数（Linear）
│       隐藏层：ReLU
│
├── 二分类
│   └── 输出层：Sigmoid
│       隐藏层：ReLU
│
├── 多分类
│   └── 输出层：CrossEntropyLoss（内置 Softmax）
│       隐藏层：ReLU
│
└── 追求极致精度 / Transformer
    ├── 隐藏层用 GELU（NLP）或 Swish/Mish（CV）
    └── 输出层同上
```

---

## 五、实战对比：7 种激活函数在糖尿病数据集上的表现

代码文件：[8. Multiple_Dimentional_Logistic_Regression.py](8.%20Multiple_Dimentional_Logistic_Regression.py)

| 激活函数 | 训练 Loss（8 轮后） | 测试准确率 | 评价 |
|:--------:|:-------------------:|:----------:|------|
| Sigmoid | ~0.69 | ~65% | 梯度消失明显，收敛慢 |
| Tanh | ~0.65 | ~67% | 略优于 Sigmoid |
| ReLU | ~0.57 | ~72% | 速度快，效果不错 |
| Leaky ReLU | ~0.56 | ~73% | 比 ReLU 略好 |
| ELU | ~0.58 | ~72% | 与 ReLU 接近，更稳定 |
| SELU | ~0.60 | ~71% | 小数据集优势不显 |
| Softplus | ~0.66 | ~68% | ReLU 的平滑版但偏慢 |

> 💡 **结论**：在这个小规模任务上，ReLU/Leaky ReLU 是性价比最高的选择。Sigmoid/Tanh 的梯度消失问题在仅 3 层网络中就已显现。

---

## 六、常见问题 FAQ

### Q1: 为什么 ReLU 允许输出为 0？输出为 0 不是浪费了吗？
不浪费。这叫做**稀疏激活**——就像人脑中只有少数神经元同时放电。强制稀疏性反而是一种隐式的正则化，让网络学到更本质的特征。

### Q2: 中间层可以用 Sigmoid 吗？
非常不推荐。深层网络用 Sigmoid 几乎一定会梯度消失。唯一例外是 LSTM/GRU 的门控机制（它们内部用 Sigmoid 做 0~1 开关，但那是刻意设计且有额外路径绕过）。

### Q3: GELU 比 ReLU 好，为什么不所有地方都用 GELU？
计算量：GELU 涉及 `tanh` 近似或 `erf` 函数，比 ReLU 的 `max(0, x)` 慢很多。对于大多数 CNN/MLP，用 ReLU 就够了，性价比更高。

### Q4: 激活函数可以混合使用吗？
当然可以。一个网络的不同层可以用不同的激活函数（比如中间层用 ReLU，最后一层用 Sigmoid）。甚至同一层的不同分支也可以用不同激活函数（如 Inception 模块）。

---

## 参考

- [PyTorch 激活函数文档](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [GELU 论文](https://arxiv.org/abs/1606.08415)
- [Swish 论文](https://arxiv.org/abs/1710.05941)
- [Mish 论文](https://arxiv.org/abs/1908.08681)
