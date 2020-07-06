# @Reference
# @Time : ${DATE} ${TIME}
# @Author : henu_bigdata
# @File : FocalBoost.py
# @Description : a classifier for imbalance data based on GPU
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier


class AdaBoostM1:

    def __init__(self):
        self.clf = None
        print('AdaBoostM1.AdaBoostM1')

    def softmax(self, x):
        """softmax"""
        xx = x - np.max(x)
        e_x = np.exp(xx)
        s = e_x / e_x.sum()

        return s

    # 适用于y为从1开始的连续标签数据
    def focal(self, nrows, ncls, y_pre, y):

        one_hot_pre = np.zeros((nrows, ncls))
        for i in range(nrows):
            col = y_pre[i] - 1
            one_hot_pre[i][col] = 1
        logit = self.softmax(one_hot_pre)
        one_hot_key = np.zeros((nrows, ncls))
        for i in range(nrows):
            col = y[i]-1
            one_hot_key[i][col] = 1

        pt = (one_hot_key * logit).sum(1)
        logpt = np.log(pt + 0.000001)
        loss = -1 * np.power((1-pt), 2) * logpt

        return loss.mean()

    def create_classifer(self, index=0):
        return RandomForestClassifier(n_estimators=20)

    def resample(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        result_arr = np.searchsorted(t, np.random.rand(len(weights))*s)
        return result_arr

    def train(self, x, y):
        y = np.array(y, dtype='int64')
        num_rows = len(x)
        ncls = len(np.unique(y))
        classifiers = []
        alphas = []
        weights = np.ones(num_rows) * 1.0 / num_rows
        for n in range(self.num_rounds):
            random_indices = self.resample(weights)
            resampled_entries_x = []
            resampled_entries_y = []
            for i in range(num_rows):
                resampled_entries_x.append(x[random_indices[i]])
                resampled_entries_y.append(y[random_indices[i]])

            weak_classifier = self.create_classifer()
            weak_classifier.fit(resampled_entries_x, resampled_entries_y, sample_weight=weights)

            # training and calculate the rate of error
            classifications = weak_classifier.predict(x)

            error = self.focal(num_rows, ncls, classifications, y) * np.array(weights)
            alpha = 0.5 * np.sum(error)

            alphas.append(alpha)
            classifiers.append(weak_classifier)

            for i in range(num_rows):
                if np.size(y[i]) > 1:
                    ry = np.argmax(y[i])
                else:
                    ry = y[i]
                h = classifications[i]
                h = (-1 if h == 0 else 1)
                ry = (-1 if ry == 0 else 1)
                weights[i] = weights[i] * np.exp(-alpha * h * ry)
            sum_weights = sum(weights)
            normalized_weights = [float(w) / sum_weights for w in weights]
            weights = normalized_weights

        self.clf = zip(alphas, classifiers)

    def predicted(self, x):

        result_list = []
        weight_list = []

        for (weight, classifier) in self.clf:
            res = classifier.predict(x)
            result_list.append(res)
            weight_list.append(weight)

        res =[]

        for i in range(len(result_list[0])):
            result_map = {}
            for j in range(len(result_list)):
                if not str(result_list[j][i]) in result_map:
                    result_map[str(result_list[j][i])] = 0
                result_map[str(result_list[j][i])] = result_map[str(result_list[j][i])] + weight_list[j]

            cur_max_value = -100000
            max_key = ''
            for key in result_map:
                if result_map[key] > cur_max_value:
                    cur_max_value = result_map[key]
                    max_key = key

            res.append(float(max_key))
        return np.asarray(res)

    def fit(self, x_train, y_train, n_rounds=20):
        self.num_rounds = n_rounds
        self.train(x_train, y_train.ravel())

    def predict(self, x_test):
        pre_label = (self.predicted(x_test)).reshape(-1, 1)
        pre_label = np.squeeze(pre_label)
