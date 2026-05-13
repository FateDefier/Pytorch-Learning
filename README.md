# PyTorch 学习笔记

根据 B站刘二大人的 [《PyTorch深度学习实践》](https://www.bilibili.com/video/BV1Y7411d7Ys) 课程编写的个人学习代码，从零基础逐步深入到经典神经网络架构。

## 学习路线

| # | 文件 | 知识点 | 数据集 |
|---|------|--------|--------|
| 1 | `Linear_model(Numpy).py` | 线性回归、NumPy 实现、MSE 损失、3D 可视化 | 合成数据 y=2x |
| 2 | `Linear_model(Torch).py` | PyTorch 线性回归、nn.Module、MSELoss、SGD | 合成数据 y=2x |
| 3 | `Stochastic_Gradient_Decent.py` | 随机梯度下降（手动实现） | 合成数据 y=2x |
| 4 | `Batch_Gradient_Decent.py` | 批量梯度下降（手动实现） | 合成数据 y=2x |
| 5 | `Back_Propagation.py` | 反向传播、autograd、计算图 | 合成数据 y=2x |
| 6 | `BP_Practice.py` | 反向传播实战——二次模型 y = w1\*x² + w2\*x + b | 合成数据 y=2x |
| 7 | `Logistic_Regression(Classification).py` | 二分类、Sigmoid、BCELoss | 合成数据 |
| 8 | `Multiple_Dimentional_Logistic_Regression.py` | 多维逻辑回归、7 种激活函数对比 | 糖尿病数据集 |
| 9 | `Softmax_Classifier(Multi-class).py` | 多分类、Softmax、CrossEntropyLoss、5 层全连接网络 | MNIST |
| 10 | `Dataset_and_DataLoader.py` | DataSet 自定义、DataLoader 小批量训练 | 糖尿病数据集 |
| 11 | `Convolutional_Neural_Network.py` | 卷积神经网络、池化、GPU 迁移 | MNIST |
| 12 | `Residual_Net.py` | 残差网络、跳跃连接、ResidualBlock | MNIST |
| 13 | `GoogleNet(Inception).py` | Inception 模块、1x1 卷积降维、多尺度特征 | MNIST |
| 14 | `Recurrent_Neural_Network.py` | 循环神经网络、RNNCell/RNN、Embedding 层 | "hello" 序列预测 |
| 15 | `Gated_Recurrent_Unit.py` | 双向 GRU、PackedSequence、文本分类 | 人名国籍数据集 |

## 模型架构速览

- **全连接网络**: `Linear → ReLU → ... → Linear` (MNIST 分类, 约 97% 准确率)
- **CNN**: `Conv → ReLU → Pool → Conv → ReLU → Pool → FC` (MNIST, 约 98%)
- **ResNet**: `Conv → Pool → ResBlock → Conv → Pool → ResBlock → FC` (MNIST, 约 99%)
- **GoogLeNet**: `Conv → Pool → Inception → Conv → Pool → Inception → FC` (MNIST, 约 98%)
- **RNN**: `Embedding → RNN → Linear` (序列预测)
- **Bi-GRU**: `Embedding → BiGRU → Linear` (人名国籍分类)

## 参考文档

- `Activation function.md` — 10 种激活函数对照表（公式、优缺点、选型建议）
- `Different Optimizer.txt` — 8 种优化器对比实验与结果分析
- `DataLoader_Result.txt` — 数据加载器训练日志

## 环境要求

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- pandas

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/xiu-yuan-siu/Pytorch-Learning.git
cd Pytorch-Learning

# 建议按文件编号顺序学习：从 1 到 15
python "Linear_model(Numpy).py"
```

部分代码需要额外数据文件（`diabetes.csv`、`names_train.csv.gz` 等），请放在项目根目录或 `../data/` 下。

## 致谢

感谢刘二大人的精彩教程，深入浅出地讲解了 PyTorch 深度学习的核心概念与实践方法。
