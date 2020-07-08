from collections import defaultdict

import numpy as np

from func_base_serial.Tree_CART import CART


class RandomForest:
    def __init__(self, n_tree=30, n_fea=None, ri_rc=True, L=None, epsilon=1e-3, min_sample=1):
        self.n_tree = n_tree
        self.n_fea = n_fea  
        self.ri_rc = ri_rc  
        self.L = L 
        self.tree_list = []  

        self.epsilon = epsilon
        self.min_sample = min_sample 

        self.D = None  
        self.N = None

    def init_param(self, X_data):
        # 
        self.D = X_data.shape[1]
        self.N = X_data.shape[0]
        if self.n_fea is None:
            self.n_fea = int(np.log2(self.D) + 1) 
        return

    def extract_fea(self):
        # 
        if self.ri_rc:
            if self.n_fea > self.D:
                raise ValueError('the number of features should be lower than dimention of data while RI is chosen')
            fea_arr = np.random.choice(self.D, self.n_fea, replace=False)
        else:
            fea_arr = np.zeros((self.n_fea, self.D))
            for i in range(self.n_fea):
                out_fea = np.random.choice(self.D, self.L, replace=False)
                coeff = np.random.uniform(-1, 1, self.D) 
                coeff[out_fea] = 0
                fea_arr[i] = coeff
        return fea_arr

    def extract_data(self, X_data, y_data):
        # 
        fea_arr = self.extract_fea()  # col_index or coeffs
        inds = np.unique(np.random.choice(self.N, self.N))  # row_index, 
        sub_X = X_data[inds]
        sub_y = y_data[inds]
        if self.ri_rc:
            sub_X = sub_X[:, fea_arr]
        else:
            sub_X = sub_X @ fea_arr.T
        return sub_X, sub_y, fea_arr

    def fit(self, X_data, y_data):
        # 
        self.init_param(X_data)
        for i in range(self.n_tree):
            sub_X, sub_y, fea_arr = self.extract_data(X_data, y_data)
            subtree = CART(epsilon=self.epsilon, min_sample=self.min_sample)
            subtree.fit(sub_X, sub_y)
            self.tree_list.append((subtree, fea_arr))  # 
        return

    def predict(self, X):
        # 
        res = defaultdict(int)  # 
        for item in self.tree_list:
            subtree, fea_arr = item
            if self.ri_rc:
                X_modify = X[fea_arr]
            else:
                X_modify = (np.array([X]) @ fea_arr.T)[0]
            label = subtree.predict(X_modify)
            res[label] += 1
        return max(res, key=res.get)

    def predicted(self, test):
        labels = []
        for i in test:
            labels.append(self.predict(i))
        return labels
