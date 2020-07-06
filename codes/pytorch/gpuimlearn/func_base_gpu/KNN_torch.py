from torchvision import datasets, transforms
import numpy as np
import torch
from torch.autograd import Variable


class KNN_torch:

    def __init__(self, k=5):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def predict(self, x_test):

        self.x_test = np.array(x_test)

        m = len(self.x_test)
        n = len(self.x_train)

        # cal Eud distance mat
        xx = (self.x_test ** 2).sum(dim=1, keepdim=True).expand(m, n)
        yy = (self.x_train ** 2).sum(dim=1, keepdim=True).expand(n, m).transpose(0, 1)

        dist_mat = xx + yy - 2 * self.x_test.matmul(self.x_train.transpose(0, 1))
        mink_idxs = dist_mat.argsort(dim=-1)

        res = []
        for idxs in mink_idxs:
            # voting
            res.append(np.bincount(np.array([self.y_train[idx] for idx in idxs[:self.k]])).argmax())

        return np.array(res)
