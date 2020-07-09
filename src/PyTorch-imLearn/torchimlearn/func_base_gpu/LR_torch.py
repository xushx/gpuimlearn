import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LogisticRegression(nn.Module):

    def __init__(self, nf=2):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(nf, 1)  
        self.sm = nn.Sigmoid()   

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

        # 
        self.criterion = nn.BCELoss()
        # self.criterion=nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)  

        for epoch in range(self.iter):
            if torch.cuda.is_available():
                x_data = Variable(torch.FloatTensor(x_train)).cuda()
                y_data = Variable(torch.FloatTensor(y_train)).cuda()
            else:
                x_data = Variable(x_train)
                y_data = Variable(y_train)

            out = self.model(x_data)  

            loss = self.criterion(out, y_data) 
            print_loss = loss.data.item() 
            mask = out.ge(0.5).float()  
            correct = (mask == y_data).sum()  
            acc = correct.item()/x_data.size(0)  
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x_test):

        self.x_test = Variable(torch.FloatTensor(x_test)).cuda()

        res = []
        for i in self.x_test:
            res.append(self.model(i))

        return np.array(res)

