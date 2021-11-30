import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torch.utils.data import DataLoader # DataLoader需实例化，用于加载数据


class JudgementModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden2_dim, output_dim):
        super(JudgementModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        hidden = self.linear1(x)
        activate = torch.relu(hidden)
        hidden2 = self.linear2(activate)
        hidden3 = self.linear3(torch.relu(hidden2))
        # 注意：整个模型结构的最后一层是线性全连接层，并非是sigmoid层，是因为之后直接接CrossEntropy()损失函数，已经内置了log softmax层的过程了
        # 若损失函数使用NLLLoss()则需要在模型结构中先做好tanh或者log_softmax
        # 即：y^ = softmax(x), loss = ylog(y^) + (1-y)log(1-y^)中的过程
        output = torch.sigmoid(hidden3)
        return output


def prepare_data(size: int, rate: float):
    x_data = None
    y_data = []
    for i in range(size):
        if random.random() >= rate:
            video_index1 = random.randint(1, 11)
            video_index2 = video_index1
            y_data.append(np.int64(0))
        else:
            video_index1 = random.randint(1, 11)
            video_index2 = random.randint(1, 11)
            y_data.append(np.int64(1))

        embed1 = np.load('./database/' + str(video_index1) + '_embedding.npy')
        embed2 = np.load('./database/' + str(video_index2) + '_embedding.npy')

        frame1 = embed1[random.randint(0, embed1.shape[0] - 1)]
        frame2 = embed2[random.randint(0, embed2.shape[0] - 1)]
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

    # 下面两个魔术方法比较好写，直接照着这个格式写就行了
    def __getitem__(self, index):  # 参数index必写
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length  # 只需返回数据集的长度即可


def get_acc(outputs, labels):
    """计算acc"""
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num

    return acc


def train(epoch_num: int, judgement_model):
    x, y = prepare_data(5000, 0.5)
    my_dataset = MyDataset(x, y)
    # 实例化
    train_loader = DataLoader(dataset=my_dataset,  # 要传递的数据集
                              batch_size=32,  # 一个小批量数据的大小是多少
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
            print(f'epoch:{epoch}, num: {i+1}, loss:{loss/y_pred.shape[0]}')
            # 3. 反向传播
            loss.backward()
            # 4. 权重/模型更新
            optimizer.step()
            # 5. 梯度清零
            optimizer.zero_grad()


model = JudgementModel(1000, 512, 128, 2)
train(10, model)
