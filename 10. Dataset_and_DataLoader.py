"""
DataLoader: 支持索引和知道数据长度，进行小批量数据集的生成
Mini-Batch: 兼顾性能(SGD)和时间(GD)
Epoch: 所有样本进行了一次 forward（前馈）
Batch_Size: 每次训练（前馈、反馈、更新）使用的样本数量
Iteration: epoch分了多少个 Batch ( = total / Batch_Size)
语法：
# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
"""
# import torch
# from torch.utils.data import Dataset  # 抽象类，不能被实例化，只能被其他类继承
# from torch.utils.data import DataLoader
#
#
# class DiabetesDataset(Dataset):
#     def __init__(self):
#         pass
#
#     def __getitem__(self, index):  # 获取索引
#         pass
#
#     def __len__(self):  # 获取长度，魔法函数
#         pass
#
#
# dataset = DiabetesDataset()
# # num_workers：读取样本的时候是否多进程，当前为 2 核，本电脑为 16 核，建议使用 15 核
# train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
# ......
# if __name__ == '__main__':  # 解决多线程遇到的问题
#     for epoch in range(100):
#         for i, data in enumerate(train_loader, 0):
#             ...

import torch
import pandas as pd
from torch.utils.data import Dataset  # 抽象类，不能被实例化，只能被其他类继承
from torch.utils.data import DataLoader


class DiabetesDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('diabetes.csv')
        self.len = df.shape[0]
        self.x_np = df.iloc[:, :-1].values
        self.y_np = df.iloc[:, [-1]].values
        self.x_data = torch.tensor(self.x_np, dtype=torch.float32)
        self.y_data = torch.tensor(self.y_np, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  # 返回元组，可拆包

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,  batch_size=32, shuffle=True, num_workers=15)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()  # 来自 Module

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(10):
        # 写法一：for i, (inputs, labels) in enumerate(train_loader, 0):不用inputs, labels = data这一行
        # 写法二
        for i, data in enumerate(train_loader, 0):
            # 1. Prepare data
            inputs, labels = data
            # 2. Forward
            y_hat = model(inputs)
            loss = criterion(y_hat, labels)
            print(epoch, i, loss.item())
            # 3. Backward
            optimizer.zero_grad()
            # 4. Update
            optimizer.step()

"""
输出结果：
0 0 0.651732325553894
0 1 0.7105777859687805
0 2 0.6531586050987244
0 3 0.6755908727645874
0 4 0.6443730592727661
0 5 0.6367462873458862
0 6 0.6518690586090088
0 7 0.6525323987007141
0 8 0.65897136926651
0 9 0.6527538299560547
0 10 0.6953524351119995
0 11 0.6613678336143494
0 12 0.6526027917861938
0 13 0.6859689950942993
0 14 0.7097391486167908
0 15 0.6581535339355469
0 16 0.6525601744651794
0 17 0.64215487241745
0 18 0.6183272004127502
0 19 0.627021849155426
0 20 0.6595266461372375
0 21 0.686245858669281
0 22 0.651798665523529
0 23 0.6926921606063843
1 0 0.6684944033622742
1 1 0.6353259086608887
1 2 0.6175576448440552
1 3 0.6703225374221802
1 4 0.677198052406311
1 5 0.6594446897506714
1 6 0.642396867275238
1 7 0.6769471764564514
1 8 0.6515628099441528
1 9 0.6688223481178284
1 10 0.6527652740478516
1 11 0.7279613018035889
1 12 0.668936014175415
1 13 0.6946542263031006
1 14 0.6837055683135986
1 15 0.6427007913589478
1 16 0.6367256045341492
1 17 0.6518846750259399
1 18 0.6538523435592651
1 19 0.6418195962905884
1 20 0.6440284848213196
1 21 0.6519103050231934
1 22 0.7116996049880981
1 23 0.6511015295982361
2 0 0.6761292815208435
2 1 0.6917481422424316
2 2 0.6605433225631714
2 3 0.6790494322776794
2 4 0.6285481452941895
2 5 0.6686849594116211
2 6 0.6762899160385132
2 7 0.645219087600708
2 8 0.7003686428070068
2 9 0.6482548713684082
2 10 0.6592831015586853
2 11 0.6439157724380493
2 12 0.6356558799743652
2 13 0.6940751075744629
2 14 0.6377909779548645
2 15 0.6632077693939209
2 16 0.6934786438941956
2 17 0.6606283187866211
2 18 0.6274895668029785
2 19 0.678329586982727
2 20 0.6502935886383057
2 21 0.6429567933082581
2 22 0.6767801642417908
2 23 0.6430965065956116
3 0 0.6699909567832947
3 1 0.6676292419433594
3 2 0.6447489261627197
3 3 0.6774739027023315
3 4 0.6597779989242554
3 5 0.667782187461853
3 6 0.6420140862464905
3 7 0.6432569026947021
3 8 0.6534990072250366
3 9 0.6611116528511047
3 10 0.6852524280548096
3 11 0.6662545204162598
3 12 0.6699190735816956
3 13 0.6850241422653198
3 14 0.7012656927108765
3 15 0.6603577136993408
3 16 0.6445341110229492
3 17 0.6530358791351318
3 18 0.6690036654472351
3 19 0.6769530773162842
3 20 0.6603351831436157
3 21 0.6095682978630066
3 22 0.6685092449188232
3 23 0.6445198655128479
4 0 0.7005934715270996
4 1 0.6442341208457947
4 2 0.669144868850708
4 3 0.6764245629310608
4 4 0.636598527431488
4 5 0.6450872421264648
4 6 0.6625510454177856
4 7 0.6443047523498535
4 8 0.6452625393867493
4 9 0.6856867671012878
4 10 0.6450193524360657
4 11 0.6609622240066528
4 12 0.633219301700592
4 13 0.6351364254951477
4 14 0.6931685209274292
4 15 0.6768361330032349
4 16 0.6428488492965698
4 17 0.6506569981575012
4 18 0.700046718120575
4 19 0.6778411269187927
4 20 0.6829842925071716
4 21 0.6858259439468384
4 22 0.6339623928070068
4 23 0.6534214019775391
"""
# titanic数据练习
