"""
ASCII码：为保证长度一致，按最长的对其他长度不足的添 0
GRU:
input: [batch_size, seq_len]
embedding后: [seq_len, batch_size, hidden_size]
Bi-directional RNN/GRU/LSTM: 最终的输出 h_N 作为 h_0(backward，不是反向传播),两层 RNN 做拼接(concatenate), hidden = [h_N^f, h_N^b]
"""
import time
import math
import torch
import gzip
import csv
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = False


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


def name2list(name):  # 2 = to
    arr = [ord(c) for c in name]
    return arr, len(arr)


def make_tensor(names, countries):
    sequence_and_lengths = [name2list(name) for name in names]
    name_sequence = [sl[0] for sl in sequence_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_lengths])
    countries = countries.long()

    # make tensor of name, Batch_size x Seq_len
    seq_tensor = torch.zeros(len(name_sequence), seq_lengths.max()).long()  # 先构造全零张量，再粘贴
    for idx, (seq, seq_len) in enumerate(zip(name_sequence, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


class NameDataset(Dataset):
    def __init__(self, is_train_set=True):  # 参数判断记载训练集还是数据集
        filename = '../data/names_train.csv.gz' if is_train_set else '../data/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:  # 使用 gzip 和 csv 包读取
            reader = csv.reader(f)
            rows = list(reader)  # (names, languages)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))  # 先列表变集合，去除重复
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        country = self.country_dict[self.countries[index]]
        return self.names[index], country  # names: string, countries: index

    def __len__(self):
        return self.len

    def getCountryDict(self):  # 构建字典：key(names), values(index)
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def getCountryNum(self):
        return self.country_num


train_set = NameDataset(is_train_set=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = NameDataset(is_train_set=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = train_set.getCountryNum()  # output_size


class RNNClassifier(torch.nn.Module):
    # input_size(seq_len, batch_size) output_size(seq_len, batch_size, hidden_size)
    # bidirectional: 单向还是双向，下面是它的输入输出
    # inputs: input(seq_len, batch_size, hidden_size), hidden(n_layers*n_directions, batch_size, hidden_size)
    # outputs: output(seq_len, batch_size, hidden_size*n_directions), hidden(n_layers*n_directions, batch_size, hidden_size)
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size  # for GRU
        self.n_layers = n_layers  # For GRU
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # input shape: B x S --> S x B（embedding需要）
        input = input.t()  # 转置
        batch_size = input.size(1)

        hidden = self.init_hidden(batch_size)
        embedding = self.embedding(input)

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # 减少计算，要先按length从大到小排序，去除padded的元素

        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(train_loader, 1):
        inputs, seq_lengths, target = make_tensor(names, countries)  # 输入，每个的长度，标签
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f"[{time_since(start)}] Epoch {epoch} ", end='')
            print(f"[{i * len(inputs)} / {len(train_set)}] ", end='')
            print(f"loss={total_loss / (i * len(inputs))}")
    return total_loss

def testModel():
    correct = 0
    total = len(test_set)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(test_loader, 1):
            inputs, seq_lengths, target = make_tensor(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f"Test set: Accuracy: {correct / total} {percent}%")
    return correct / total

if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device('cuda:0')
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    # 可视化
    epochs = np.arange(1, N_EPOCHS + 1)
    accuracies = np.array(acc_list)

    # 找最大 Accuracy 及其位置
    max_acc = accuracies.max()
    max_epoch = epochs[accuracies.argmax()]

    plt.figure()
    plt.plot(epochs, accuracies, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")

    # 标出最大点
    plt.scatter(max_epoch, max_acc)
    plt.annotate(
        f"({max_epoch}, {max_acc:.4f})",
        xy = (max_epoch, max_acc),
        xytext=(max_epoch, max_acc),
        textcoords="offset points",
        xycoords='data'
    )

    plt.show()
"""
应用：评论分类，生成古诗、文言文，生成Linux源码
"""
