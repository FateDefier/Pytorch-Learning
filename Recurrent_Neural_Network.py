"""
RNN(Recurrent Neural Network)本质上是线性元的复用，每次的输出结果作为下一次线性元的输入参数
RNNCell: 要自己手写循环进行遍历
例1：
cell = torch.nn.RNNCell(input_size, hidden_size, num_layers, bidirectional=True)
hidden = cell(input, hidden)
input:
1.input of shape (batch, input_size)
2.hidden of shape (batch, hidden_size)
output:
1.hidden of shape (batch, hidden_size)
例 2:
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seq, batch, features)
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):  # idx = seq_len,即序列索引号
    print('=' * 20, idx, '=' * 20)
    print('Input size: ', input.shape)

    hidden = cell(input, hidden)

    print('output size: ', hidden.shape)
    print(hidden)
输出结果：
==================== 0 ====================
Input size:  torch.Size([1, 4])
output size:  torch.Size([1, 2])
tensor([[-0.4153,  0.4558]], grad_fn=<TanhBackward0>)
==================== 1 ====================
Input size:  torch.Size([1, 4])
output size:  torch.Size([1, 2])
tensor([[-0.6762, -0.8070]], grad_fn=<TanhBackward0>)
==================== 2 ====================
Input size:  torch.Size([1, 4])
output size:  torch.Size([1, 2])
tensor([[-0.1584, -0.4740]], grad_fn=<TanhBackward0>)

RNN: 自己自动循环
1.batch_first = True参数：将来提供参数时把 batch_Size 和 seqLen 交换位置，即提供的数据 (batchSize, seqLen, input_size)
例 1 ：
num_layers：RNN 的 hidden_layer 的层数
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
out, hidden = cell(inputs, hidden)
inputs: x1, x2, x3...xN (shape (seqSize, batch, input_size))
hidden(input): h0       (shape (num_layers, batch, hidden_size))
out: h1, h2, h3...hN    (shape (seqSize, batch, hidden_size))
hidden(output): hN      (shape (num_layers, batch, hidden_size))
例 2 ：
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 2

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# (seq, batch, features)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print("Output size: ", out.shape)
print("Output: ", out)
print("Hidden size: ", hidden.shape)
print("Hidden: ", hidden)
输出结果：
Output size:  torch.Size([3, 1, 2])
Output:  tensor([[[ 0.1581,  0.7508]],

        [[-0.1614,  0.9469]],

        [[-0.3630,  0.9082]]], grad_fn=<StackBackward0>)
Hidden size:  torch.Size([2, 1, 2])
Hidden:  tensor([[[-0.9033,  0.2967]],

        [[-0.3630,  0.9082]]], grad_fn=<StackBackward0>)
例 3 （使用 batch_first 参数）：
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 2

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

# (seq, batch, features)
inputs = torch.randn(batch_size, seq_len, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print("Output size: ", out.shape)
print("Output: ", out)
print("Hidden size: ", hidden.shape)
print("Hidden: ", hidden)
输出结果：
Output size:  torch.Size([1, 3, 2])
Output:  tensor([[[ 0.6292, -0.3800],
         [ 0.1840, -0.6198],
         [ 0.3127, -0.7383]]], grad_fn=<TransposeBackward1>)
Hidden size:  torch.Size([2, 1, 2])
Hidden:  tensor([[[ 0.9291,  0.2084]],

        [[ 0.3127, -0.7383]]], grad_fn=<StackBackward0>)

写 RNN 最重要的是注意 张量的维度
激活函数用 tanh

One-hot（独热向量）的缺点：
1.维度高
2.稀疏
3.硬编码，非学习得到
改进：使用 嵌入层 (Embedding): 把高维、稀疏的样本映射到低维、稠密的样本 --> 降维
"""
import torch
# 使用 RNNCell
# seq_len = 5
# batch_size = 1
# input_size = 4
# hidden_size = 4
#
# idx2char = ['e', 'h', 'l', 'o']
# x_data = [1, 0, 2, 2, 3]
# y_data = [3, 1, 2, 3, 2]
#
# one_hot_lookup = [[1, 0, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]
#
# x_one_hot = [one_hot_lookup[x] for x in x_data]  # seq_len x input_size 的列表
#
# inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# labels = torch.LongTensor(y_data).view(-1, 1)
#
#
# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size):
#         super(Model, self).__init__()
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
#
#     def forward(self, input, hidden):
#         hidden = self.rnncell(input, hidden)  # h_t = cell(x_t-1, h_t-1)
#         return hidden
#
#     def init_hidden(self):  # 生成默认的全零的 h_0，batch_size 在这里用
#         return torch.zeros(self.batch_size, self.hidden_size)
#
#
# net = Model(input_size, hidden_size, batch_size)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
#
# for epoch in range(15):
#     loss = 0
#     optimizer.zero_grad()
#     hidden = net.init_hidden()
#     print("Predicted String: ", end='')
#     for input, label in zip(inputs, labels):
#         hidden = net(input, hidden)
#         loss += criterion(hidden, label)  # 这里的 loss 是张量，相加时构建计算图，不要用 item()
#         _, idx = hidden.max(dim=1)
#         print(idx2char[idx.item()], end='')
#     loss.backward()
#     optimizer.step()
#     print(", Epoch [%d/15] loss=%.4f" % (epoch+1, loss.item()))

"""
输出结果：
Predicted String: ooooo, Epoch [1/15] loss=6.3611
Predicted String: ooooo, Epoch [2/15] loss=5.6038
Predicted String: ohooo, Epoch [3/15] loss=5.1055
Predicted String: ohooo, Epoch [4/15] loss=4.7104
Predicted String: ohooo, Epoch [5/15] loss=4.4621
Predicted String: ohooo, Epoch [6/15] loss=4.2516
Predicted String: ohllo, Epoch [7/15] loss=4.0227
Predicted String: ohllo, Epoch [8/15] loss=3.7955
Predicted String: ohlll, Epoch [9/15] loss=3.5899
Predicted String: ohlol, Epoch [10/15] loss=3.4179
Predicted String: ohlol, Epoch [11/15] loss=3.2279
Predicted String: ohlol, Epoch [12/15] loss=2.9565
Predicted String: ohlol, Epoch [13/15] loss=2.5778
Predicted String: ohlol, Epoch [14/15] loss=2.4678
Predicted String: ohlol, Epoch [15/15] loss=2.3624
"""

# 使用 RNN
# seq_len = 5
# input_size = 4
# hidden_size = 4
# batch_size = 1
# num_layers = 1
#
# idx2char = ['e', 'h', 'l', 'o']
# x_data = [1, 0, 2, 2, 3]
# y_data = [3, 1, 2, 3, 2]
#
# one_hot_lookup = [[1, 0, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]
#
# x_one_hot = [one_hot_lookup[x] for x in x_data]  # seq_len x input_size 的列表
#
# inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)  # 注意这里dim=1要变成 seq_len
# labels = torch.LongTensor(y_data)
#
#
# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size, num_layers):
#         super(Model, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
#
#     def forward(self, input):
#         hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
#         out, _ = self.rnn(input, hidden)
#         return out.view(-1, self.hidden_size)
#
#
# net = Model(input_size, hidden_size, batch_size, num_layers)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
#
# for epoch in range(15):
#     optimizer.zero_grad()
#     outputs = net(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#
#     _, idx = torch.max(outputs, 1)
#     idx = idx.data.numpy()
#     print("Predicted String: ", ''.join([idx2char[x] for x in idx]), end='')
#     print(", epoch [%d/15] loss: %.4f" % (epoch+1, loss.item()))

"""
输出结果:
Predicted String:  lolle, epoch [1/15] loss: 1.4485
Predicted String:  lolll, epoch [2/15] loss: 1.2752
Predicted String:  lolll, epoch [3/15] loss: 1.1442
Predicted String:  lolll, epoch [4/15] loss: 1.0492
Predicted String:  lolll, epoch [5/15] loss: 0.9765
Predicted String:  oolol, epoch [6/15] loss: 0.9165
Predicted String:  oolol, epoch [7/15] loss: 0.8617
Predicted String:  oolol, epoch [8/15] loss: 0.8075
Predicted String:  oolol, epoch [9/15] loss: 0.7542
Predicted String:  oolol, epoch [10/15] loss: 0.7047
Predicted String:  ohlol, epoch [11/15] loss: 0.6613
Predicted String:  ohlol, epoch [12/15] loss: 0.6239
Predicted String:  ohlol, epoch [13/15] loss: 0.5912
Predicted String:  ohlol, epoch [14/15] loss: 0.5622
Predicted String:  ohlol, epoch [15/15] loss: 0.5361
"""

# 3.使用 嵌入层 embedding
"""
torch.nn.Embedding(num_embeddings, embedding_dim)
num_embeddings(int): size of dictionary of embedding  # 独热向量的维度
embedding_size(int): size of each embedding vector
input: LongTensor，如 (seq_len, batch_size)
Output: (*, embedding_dim), 则对应输出 (seq_len, batch_size, embedding_dim)
"""
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]  # (batch, seq_len)
y_data = [3, 1, 2, 3, 2]    # (batch * seq_len

inputs = torch.LongTensor(x_data)  # 注意: Embedding 的输入为 LongTensor
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)


net = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = torch.max(outputs, 1)
    idx = idx.data.numpy()
    print("Predicted String: ", ''.join([idx2char[x] for x in idx]), end='')
    print(", epoch [%d/15] loss: %.4f" % (epoch+1, loss.item()))

"""
输出结果：
Predicted String:  olool, epoch [1/15] loss: 1.2963
Predicted String:  ollll, epoch [2/15] loss: 1.0313
Predicted String:  ollol, epoch [3/15] loss: 0.8058
Predicted String:  ollol, epoch [4/15] loss: 0.6135
Predicted String:  ollol, epoch [5/15] loss: 0.4830
Predicted String:  ollol, epoch [6/15] loss: 0.3773
Predicted String:  ohlol, epoch [7/15] loss: 0.2958
Predicted String:  ohlol, epoch [8/15] loss: 0.2350
Predicted String:  ohlol, epoch [9/15] loss: 0.1910
Predicted String:  ohlol, epoch [10/15] loss: 0.1603
Predicted String:  ohlol, epoch [11/15] loss: 0.1368
Predicted String:  ohlol, epoch [12/15] loss: 0.1172
Predicted String:  ohlol, epoch [13/15] loss: 0.1003
Predicted String:  ohlol, epoch [14/15] loss: 0.0855
Predicted String:  ohlol, epoch [15/15] loss: 0.0721
"""

