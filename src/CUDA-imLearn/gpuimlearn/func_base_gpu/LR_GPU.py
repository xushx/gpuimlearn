# @Reference
# @Time : ${DATE} ${TIME}
# @File : LR_GPU.py
# @Description : Logistic Regression classifier based on GPU

from ctypes import *
from numpy.ctypeslib import ndpointer

import numpy as np

lr = cdll.LoadLibrary(r'gpuimlearn/func_base_gpu/LR_CUDA.so')
params = np.array([])


class LR_GPU(object):
    def __init__(self, epochs=200):
        self.epochs = epochs
        # self.params = []
        # print('LR_GPU')

    def fit(self, x_train, y_train):

        x_train = np.asarray(x_train, dtype=c_float)
        y_train = np.squeeze(np.asarray(y_train, dtype=c_float))
        # self.normalization(x_train)
        xH, xW = x_train.shape
        x_train = x_train.flatten()

        self.params = np.array(-1.0 + 2.0 * (np.random.random(xW)), dtype=c_float)

        lr.fit.argtypes = [ndpointer(c_float), ndpointer(c_float), ndpointer(c_float), c_int, c_int, c_int, c_float]
        lr.fit.restype = c_void_p
        lr.fit(x_train, y_train, self.params, c_int(xH), c_int(xW), c_int(self.epochs), c_float(0.05))

        # print(self.params)

    def predict(self, x_test):

        x_test = np.asarray(x_test, dtype=c_float)
        tH, tW = x_test.shape
        x_test = x_test.flatten()
        res = np.zeros(tH, dtype=c_float)

        lr.predicted.argtypes = [ndpointer(c_float), ndpointer(c_float), ndpointer(c_float), c_int, c_int]
        lr.predicted.restype = c_void_p
        lr.predicted(x_test, res, self.params, c_int(tH), c_int(tW))
        res = np.asarray(res, dtype='int64').flatten()
        # print(res)

        return res

    # def normalization(self, data):
    #     range = np.max(data) - np.min(data)
    #     print(range)
    #
    #     return (data - np.min(data)) / range


