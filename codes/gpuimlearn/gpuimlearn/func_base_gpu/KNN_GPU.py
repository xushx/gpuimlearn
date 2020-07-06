# @Reference
# @Time : ${DATE} ${TIME}
# @Author : henu_bigdata
# @File : KNN_GPU.py
# @Description : KNN classifier based on GPU

from collections import Counter

from ctypes import *
from numpy.ctypeslib import ndpointer

import numpy as np
from scipy.stats import mode

knn = cdll.LoadLibrary(r'gpuimlearn/func_base_gpu/KNN_CUDA.so')


class KNN_GPU(object):

    def __init__(self, k=5):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = np.asarray(x_train, dtype=c_float)
        self.x_row, self.x_col = self.x_train.shape
        self.x_train = self.x_train.flatten()
        self.y_train = np.asarray(y_train, dtype="int64").flatten()

    def predict(self, x_test):
        x_test = np.asarray(x_test, dtype=c_float)
        t_row, t_col = x_test.shape
        x_test = x_test.flatten()

        dis = np.ones(t_row*self.k, dtype=c_float)
        index = np.ones(t_row*self.k, dtype=c_int)

        knn.knn_cuda.argtypes = [ndpointer(c_float), c_int, ndpointer(c_float), c_int, c_int, c_int, ndpointer(c_float),
                                 ndpointer(c_int)]
        knn.knn_cuda.restype = c_void_p
        knn.knn_cuda(self.x_train, c_int(self.x_row), x_test, c_int(t_row), c_int(t_col), c_int(self.k),
                     dis, index)
        index = index.reshape((t_row, self.k))

        sort_labels = self.y_train[index]
        res = mode(sort_labels, axis=1)[0]

        return np.array(res)
        # print(result)
