import math
from collections import defaultdict

import numpy as np


class AdaBoost:
    def __init__(self, epsilon=0.02):
        self.epsilon = epsilon 
        self.w = None  
        self.N = None
        self.g_list = [] 
        self.alpha = []  
        self.base_list = [] 

    def init_param(self, X_data):
        # 
        self.N = X_data.shape[0]
        self.w = np.ones(self.N) / self.N  
        for i in range(1, self.N): 
            nu = (X_data[i][0] + X_data[i - 1][0]) / 2
            self.g_list.append((0, nu)) 
            self.g_list.append((1, nu))
        return

    def cal_weak_val(self, nu, X):
        # 
        val = 1
        if (nu[0] == 0 and X[0] > nu[1]) or (nu[0] == 1 and X[0] <= nu[1]):
            val = -1
        return val

    def get_base(self, X_data, y_data):
        # 
        g_err = defaultdict(float)  # 

        for g in self.g_list:
            for i in range(self.N):
                if self.cal_weak_val(g, X_data[i]) != y_data[i]:
                    g_err[g] += self.w[i]  # 

        best_g = min(g_err, key=g_err.get)
        return best_g, g_err[best_g]

    def cal_alpha(self, err):
        # 
        return 1.0 / 2 * math.log((1 - err + 0.00001) / (err + 0.00001))

    def cal_weight(self, X_data, y_data, base, alpha):
        # 
        for i in range(self.N):
            self.w[i] *= math.exp(-alpha * y_data[i] * self.cal_weak_val(base, X_data[i]))
        self.w = self.w / np.sum(self.w)
        return

    def _fx(self, X):
        # 
        s = 0
        for alpha, base in zip(self.alpha, self.base_list):
            s += alpha * self.cal_weak_val(base, X)
        return np.sign(s)

    def fit(self, X_data, y_data):
        y_data = np.squeeze(y_data)
        # 
        self.init_param(X_data)
        depth = 20
        while depth > 0:  # 
            depth -= 1
            base, err = self.get_base(X_data, y_data)
            alpha = self.cal_alpha(err)
            self.cal_weight(X_data, y_data, base, alpha)  # 
            self.alpha.append(alpha)
            self.base_list.append(base)

            s = 0
            for X, y in zip(X_data, y_data):
                if self._fx(X) != y:
                    s += 1
            if s / self.N <= self.epsilon:  # 
                print('the err ratio is {0}'.format(s / self.N))
                break
        return

    def predict(self, X):
        # 
        return self._fx(X)

    def predicted(self, test):
        labels = []
        for i in test:
            labels.append(self.predict(i))
        return labels

