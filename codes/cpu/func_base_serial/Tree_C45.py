from collections import Counter, defaultdict

import numpy as np
import time


class node:
    # 这里构建树的节点类，也可用字典来表示树结构
    def __init__(self, fea=-1, res=None, child=None):
        self.fea = fea
        self.res = res
        self.child = child  # 特征的每个值对应一颗子树，特征值为键，相应子树为值


class C45:
    def __init__(self, epsilon=1e-3, metric='C4.5'):
        self.epsilon = epsilon
        self.tree = None
        self.metric = metric

    def exp_ent(self, y_data):
        # 计算经验熵
        c = Counter(y_data)  # 统计各个类标记的个数
        ent = 0
        N = len(y_data)
        for val in c.values():
            p = val / N
            ent += -p * np.log2(p)
        return ent

    def con_ent(self, fea, X_data, y_data):
        # 计算条件熵并返回，同时返回切分后的各个子数据集
        fea_val_unique = Counter(X_data[:, fea])
        subdata_inds = defaultdict(list)  # 根据特征fea下的值切分数据集
        for ind, sample in enumerate(X_data):
            subdata_inds[sample[fea]].append(ind)  # 挑选某个值对应的所有样本点的索引

        ent = 0
        N = len(y_data)
        for key, val in fea_val_unique.items():
            pi = val / N
            ent += pi * self.exp_ent(y_data[subdata_inds[key]])
        return ent, subdata_inds

    def infoGain(self, fea, X_data, y_data):
        # 计算信息增益
        exp_ent = self.exp_ent(y_data)
        con_ent, subdata_inds = self.con_ent(fea, X_data, y_data)
        return exp_ent - con_ent, subdata_inds

    def infoGainRatio(self, fea, X_data, y_data):
        # 计算信息增益比
        g, subdata_inds = self.infoGain(fea, X_data, y_data)
        N = len(y_data)
        split_info = 1e-5
        for val in subdata_inds.values():
            p = len(val) / N
            split_info -= p * np.log2(p)
        r1 = g / split_info
        r2 = subdata_inds

        return r1, r2

    def bestfea(self, fea_list, X_data, y_data):
        # 获取最优切分特征、相应的信息增益（比）以及切分后的子数据集
        score_func = self.infoGainRatio
        if self.metric == 'ID3':
            score_func = self.infoGain
        bestfea = fea_list[0]  # 初始化最优特征
        gmax, bestsubdata_inds = score_func(bestfea, X_data, y_data)  # 初始化最大信息增益及切分后的子数据集
        for fea in fea_list[1:]:
            g, subdata_inds = score_func(fea, X_data, y_data)
            if g > gmax:
                bestfea = fea
                bestsubdata_inds = subdata_inds
                gmax = g
        return gmax, bestfea, bestsubdata_inds

    def buildTree(self, fea_list, X_data, y_data):
        # 递归构建树
        label_unique = np.unique(y_data)
        if label_unique.shape[0] == 1:  # 数据集只有一个类，直接返回该类
            return node(res=label_unique[0])
        if not fea_list:
            return node(res=Counter(y_data).most_common(1)[0][0])
        gmax, bestfea, bestsubdata_inds = self.bestfea(fea_list, X_data, y_data)
        if gmax < self.epsilon:  # 信息增益比小于阈值，返回数据集中出现最多的类
            return node(res=Counter(y_data).most_common(1)[0][0])
        else:
            fea_list.remove(bestfea)
            child = {}
            for key, val in bestsubdata_inds.items():
                child[key] = self.buildTree(fea_list, X_data[val], y_data[val])
            return node(fea=bestfea, child=child)

    def fit(self, X_data, y_data):
        y_data = np.squeeze(y_data)
        fea_list = list(range(X_data.shape[1]))
        self.tree = self.buildTree(fea_list, X_data, y_data)
        return

    def predict(self, X):
        def helper(X, tree):
            if tree.res is not None:  # 表明到达叶节点
                return tree.res
            else:
                try:
                    sub_tree = tree.child[X[tree.fea]]
                    return helper(X, sub_tree)  # 根据对应特征下的值返回相应的子树
                except:
                    # print('input data is out of scope')
                    pass

        return helper(X, self.tree)

    def predicted(self, test):
        labels = []
        for i in test:
            labels.append(self.predict(i))
        return labels
