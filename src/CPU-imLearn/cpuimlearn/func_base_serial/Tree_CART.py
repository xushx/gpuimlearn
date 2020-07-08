from collections import Counter, defaultdict

import numpy as np
import time


class node:
    def __init__(self, fea=-1, val=None, res=None, right=None, left=None):
        self.fea = fea 
        self.val = val 
        self.res = res 
        self.right = right
        self.left = left


class CART:
    def __init__(self, epsilon=1e-2, min_sample=1):
        self.epsilon = epsilon
        self.min_sample = min_sample  
        self.tree = None

    def getGini(self, y_data):
        # 
        c = Counter(y_data)
        return 1 - sum([(val / y_data.shape[0]) ** 2 for val in c.values()])

    def getFeaGini(self, set1, set2):
        # 
        num = set1.shape[0] + set2.shape[0]
        return set1.shape[0] / num * self.getGini(set1) + set2.shape[0] / num * self.getGini(set2)

    def bestSplit(self, splits_set, X_data, y_data):
        # 
        pre_gini = self.getGini(y_data)
        subdata_inds = defaultdict(list)  
        for split in splits_set:
            for ind, sample in enumerate(X_data):
                if sample[split[0]] == split[1]:
                    subdata_inds[split].append(ind)
        min_gini = 1
        best_split = None
        best_set = None
        for split, data_ind in subdata_inds.items():
            set1 = y_data[data_ind]  
            set2_inds = list(set(range(y_data.shape[0])) - set(data_ind))
            set2 = y_data[set2_inds]
            if set1.shape[0] < 1 or set2.shape[0] < 1:
                continue
            now_gini = self.getFeaGini(set1, set2)
            if now_gini < min_gini:
                min_gini = now_gini
                best_split = split
                best_set = (data_ind, set2_inds)
        if abs(pre_gini - min_gini) < self.epsilon:  
            best_split = None
        return best_split, best_set, min_gini

    def buildTree(self, splits_set, X_data, y_data):
        if y_data.shape[0] < self.min_sample: 
            return node(res=Counter(y_data).most_common(1)[0][0])
        best_split, best_set, min_gini = self.bestSplit(splits_set, X_data, y_data)
        if best_split is None:  
            return node(res=Counter(y_data).most_common(1)[0][0])
        else:
            splits_set.remove(best_split)
            left = self.buildTree(splits_set, X_data[best_set[0]], y_data[best_set[0]])
            right = self.buildTree(splits_set, X_data[best_set[1]], y_data[best_set[1]])
            return node(fea=best_split[0], val=best_split[1], right=right, left=left)

    def fit(self, X_data, y_data):

        y_data = np.squeeze(y_data)

        splits_set = []
        for fea in range(X_data.shape[1]):
            unique_vals = np.unique(X_data[:, fea])
            if unique_vals.shape[0] < 2:
                continue
            elif unique_vals.shape[0] == 2:  
                splits_set.append((fea, unique_vals[0]))
            else:
                for val in unique_vals:
                    splits_set.append((fea, val))
        self.tree = self.buildTree(splits_set, X_data, y_data)
        return

    def predict(self, x):
        def helper(x, tree):
            if tree.res is not None:  
                return tree.res
            else:
                if x[tree.fea] == tree.val:  
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x, branch)

        return helper(x, self.tree)

    def predicted(self, test):
        labels = []
        for i in test:
            labels.append(self.predict(i))
        return labels

    def disp_tree(self):

        self.disp_helper(self.tree)
        return

    def disp_helper(self, current_node):
        
        print(current_node.fea, current_node.val, current_node.res)
        if current_node.res is not None:
            return
        self.disp_helper(current_node.left)
        self.disp_helper(current_node.right)
        return
