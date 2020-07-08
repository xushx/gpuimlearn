import numpy as np
import operator
import math


class KnnClassifier:

    def __init__(self, k=5):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        self.x_test = x_test
        labels = []

        # arr_dis = self.compute_dis(self.x_train, self.x_test)
        # m, n = arr_dis.shape
        # for i in arr_dis:
        #     sort_arr = self.sort(i)
        #     larr = np.squeeze(self.y_train[sort_arr])
        #     count = np.bincount(larr)
        #     label = np.argmax(count)
        #     labels.append(label)

        for test in self.x_test:
            test_temp = np.tile(test, (len(self.y_train), 1))
            dis_temp = (test_temp - self.x_train) ** 2
            dis = dis_temp.sum(axis=1) ** 0.5
            sort_dis = dis.argsort()
            sort_labels = self.y_train[sort_dis][0:self.k]
            sort_labels = np.squeeze(sort_labels)
            count = np.bincount(sort_labels)
            label = np.argmax(count)
            labels.append(label)
        
        labels = np.asarray(labels)
        return labels

    def compute_dis(self, A, B):
        m, n = A.shape
        l, o = B.shape
        di = np.zeros((l, m))

        for i in range(l):
            for j in range(m):
                val = 0.0
                for k in range(n):
                    val += (B[i][k] - A[j][k]) * (B[i][k] - A[j][k])
                di[i][j] = math.sqrt(val)

        return di

    def sort(self, arr):
        l = len(arr)
        aux = np.zeros(l)

        for i in range(l):
            min_idx = i
            for j in range(i + 1, l):
                if arr[min_idx] > arr[j]:
                    min_idx = j
            aux[i] = min_idx
            arr[i], arr[min_idx] = arr[min_idx], arr[i]

        res = aux[0:self.k].astype('int64')
        return res


