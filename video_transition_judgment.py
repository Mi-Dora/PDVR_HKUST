import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torch.utils.data import DataLoader  # DataLoader需实例化，用于加载数据


class JudgementModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden2_dim, output_dim):
        super(JudgementModel, self).__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden2_dim)
        self.norm3 = nn.BatchNorm1d(hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = self.linear2(torch.tanh(x))
        x = self.norm3(x)
        x = self.linear3(torch.tanh(x))
        # 注意：整个模型结构的最后一层是线性全连接层，并非是sigmoid层，是因为之后直接接CrossEntropy()损失函数，已经内置了log softmax层的过程了
        # 若损失函数使用NLLLoss()则需要在模型结构中先做好tanh或者log_softmax
        # 即：y^ = softmax(x), loss = ylog(y^) + (1-y)log(1-y^)中的过程
        output = torch.sigmoid(x)
        return output


# 获取数据，从embedding数据中，rate为阴性(negative)负样本的比率
def prepare_data(size: int, rate: float):
    x_data = None
    y_data = []
    for i in range(size):
        if random.random() >= rate:
            video_index = random.randint(1, 11)
            embed = np.load('./database/' + str(video_index) + '_embedding.npy')
            frame_index = random.randint(1, embed.shape[0] - 1)
            frame1 = embed[frame_index]
            frame2 = embed[frame_index - 1]
            y_data.append(np.int64(0))
        else:
            video_index1 = random.randint(1, 11)
            video_index2 = random.randint(1, 11)
            embed1 = np.load('./database/' + str(video_index1) + '_embedding.npy')
            embed2 = np.load('./database/' + str(video_index2) + '_embedding.npy')
            frame1 = embed1[random.randint(0, embed1.shape[0] - 1)]
            frame2 = embed2[random.randint(0, embed2.shape[0] - 1)]
            y_data.append(np.int64(1))

        new_data = np.vstack((frame1[np.newaxis, :], frame2[np.newaxis, :]))[np.newaxis, ...]
        if x_data is None:
            x_data = new_data
        else:
            x_data = np.vstack((x_data, new_data))
    return x_data, np.array(y_data)


class MyDataset(Dataset):  # 继承Dataset类
    def __init__(self, x_data, y_data):
        # 把数据和标签拿出来
        x_data = x_data.reshape(x_data.shape[0], -1)

        self.x_data = x_data
        self.y_data = y_data
        # 数据集的长度
        self.length = self.y_data.shape[0]

    def __getitem__(self, index):  # 参数index必写
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length  # 只需返回数据集的长度即可

class RunDataSet(Dataset):  # 继承Dataset类
    def __init__(self, x_data):
        # 把数据拿出来
        x_data = x_data.reshape(x_data.shape[0], -1)

        self.x_data = x_data
        # 数据集的长度
        self.length = self.x_data.shape[0]

    def __getitem__(self, index):  # 参数index必写
        return self.x_data[index]

    def __len__(self):
        return self.length  # 只需返回数据集的长度即可


def get_acc(outputs, labels):
    """计算acc"""
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0] * 1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num

    return acc

'''获取precision'''
def get_pre(outputs, labels, threshold):
    TP = 0
    TN = 0
    for i in range(len(outputs)):
        if outputs[i][1] >= threshold:
            if labels[i] == 1:
                TP += 1
            else:
                TN += 1
    return TP*1.0/(TP+TN)


'''获取recall'''
def get_rec(outputs, labels, threshold):
    TP = 0
    FN = 0
    for i in range(len(outputs)):
        if labels[i] == 1:
            if outputs[i][1] >= threshold:
                TP += 1
            else:
                FN += 1
    return TP*1.0/(TP+FN)


def train(epoch_num: int, judgement_model):
    x, y = prepare_data(10000, 0.5)
    my_dataset = MyDataset(x, y)
    # 实例化
    train_loader = DataLoader(dataset=my_dataset,  # 要传递的数据集
                              batch_size=96,  # 一个小批量数据的大小是多少
                              shuffle=True,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                              num_workers=0)  # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误

    optimizer = torch.optim.SGD(judgement_model.parameters(), lr=0.01, momentum=0.9)
    # 损失
    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # 2. 前向传播
            y_pred = judgement_model(inputs)
            loss = loss_fun(y_pred, labels)
            # 3. 反向传播
            loss.backward()
            # 4. 权重/模型更新
            optimizer.step()
            # 5. 梯度清零
            optimizer.zero_grad()
            if i == 10:
                print(f'epoch:{epoch + 1}, loss:{loss}')


# judgement_model = judgement_model(1000, 512, 128, 2)
# train(100, judgement_model)
#
# torch.save(judgement_model.state_dict(), './output_data/judgement_model')
if __name__ == '__main__':
    model = JudgementModel(1000, 512, 128, 2)
    model.load_state_dict(torch.load('./output_data/judgement_model'))
    model.eval()
    x, y = prepare_data(3360, 0.5)
    test_dataset = MyDataset(x, y)
    # 实例化
    test_loader = DataLoader(dataset=test_dataset,  # 要传递的数据集
                             batch_size=3360,  # 一个小批量数据的大小是多少
                             shuffle=True,  # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                             num_workers=0)  # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误

    for i, data in enumerate(test_loader):
        inputs, labels = data
        # 2. 前向传播
        y_pred = model(inputs)


# from matplotlib import pyplot as plt
#
# plt.title('PR graph')
# plt.xlabel('recall')
# plt.ylim(0.5, 1)
# plt.ylabel('precision')
# plt.grid(linestyle='dotted')
# plt.plot(np.array(x_), np.array(y_), marker='o', markersize=1, linewidth=3)
# plt.show()