import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LogisticRegression(nn.Module):

    def __init__(self, nf=2):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(nf, 1)   #相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
        self.sm = nn.Sigmoid()   #相当于通过激活函数的变换

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


class LR_torch:

    def __init__(self, niter=200, lr=0.05):
        self.iter = niter
        self.lr = lr

    def fit(self, x_train, y_train):

        num, nf = x_train.shape
        y_train.resize((num, 1))

        self.model = LogisticRegression(nf=nf)
        if torch.cuda.is_available():
            self.model.cuda()

        # 定义损失函数和优化器
        self.criterion = nn.BCELoss()
        # self.criterion=nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)  # 采用随机梯度下降的方法

        for epoch in range(self.iter):
            if torch.cuda.is_available():
                x_data = Variable(torch.FloatTensor(x_train)).cuda()
                y_data = Variable(torch.FloatTensor(y_train)).cuda()
            else:
                x_data = Variable(x_train)
                y_data = Variable(y_train)

            out = self.model(x_data)  #根据逻辑回归模型拟合出的y值

            loss = self.criterion(out, y_data)  #计算损失函数
            print_loss = loss.data.item()  #得出损失函数值
            mask = out.ge(0.5).float()  #以0.5为阈值进行分类
            correct = (mask == y_data).sum()  #计算正确预测的样本个数
            acc = correct.item()/x_data.size(0)  #计算精度
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x_test):

        self.x_test = Variable(torch.FloatTensor(x_test)).cuda()

        res = []
        for i in self.x_test:
            res.append(self.model(i))

        return np.array(res)

