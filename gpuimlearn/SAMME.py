# @Reference
# @Time : ${DATE} ${TIME}
# @Author : henu_bigdata
# @File : SAMME.py
# @Description : a classifier for imbalance data based on GPU

import numpy as np
import time

from thundersvm import SVC
# from sklearn.tree import DecisionTreeClassifier


class SAMME(object):

    def __init__(self):
        self.clf = None
        print('SAMME.SAMME')
        self.num_rounds = 20

    def create_classifer(self, index=0):
        # return DecisionTreeClassifier()
        return SVC()

    # def process_alpha(self, index, alpha):
    #     db = dbm.open('params', 'c')
    #     key = 'alpha_'+str(index)
    #     db[key] = str(alpha)
    #     db.close()

    # def load(self, weight=[]):
    #     classifiers = []
    #     for index in range(len(weight)):
    #         classifiers.append(self.create_classifer(index))
    #     self.clf = zip(weight, classifiers)

    def resample(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        result_arr = np.searchsorted(t, np.random.rand(len(weights))*s)
        return result_arr

    def train(self, x, y):

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

            weak_classifier = self.create_classifer(n)
            weak_classifier.fit(resampled_entries_x, resampled_entries_y)

            # training and calculate the rate of error
            classifications = weak_classifier.predict(x)
            error = 0
            for i in range(len(classifications)):
                error += (classifications[i] != y[i]) * weights[i]

            if error == 0.:
                alpha = 4.0
            elif error > 0.7:
                continue  # discard classifier with error > 0.5
            else:
                # alpha = 0.5 * np.log((1 - error) / error)
                alpha = np.log((1 - error) / error + np.log(ncls - 1))

            # self.process_alpha(n, alpha)
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

    def fit(self, x_train, y_train):
        self.train(x_train, y_train)

    def predict(self, x_test):
        pre_label = (self.predicted(x_test)).reshape(-1, 1)
        pre_label = np.squeeze(pre_label)
        return pre_label

