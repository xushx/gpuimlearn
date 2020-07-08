import numpy as np

class LogisticRegression(object):
    def __init__(self, learning_rate=0.05, numIters=500):
        self.learning_rate = learning_rate
        self.numIters = numIters

    def sigmoid(self, x):
        res = 1.0/(1 + np.exp(-x))
        return res

    def fit(self, X, Y, reg=0.0, verbose=False):
        # print('slr.fit....')
        X = np.array(X)
        Y = np.array(Y)
        m, n = X.shape
        # 
        self.w = np.ones((n, 1))
        Y = Y.reshape((m, 1))
        loss_history = []
        # 
        for i in range(self.numIters):
            h = self.sigmoid(self.dot(X, self.w))
            loss = h-Y
            loss_history.append(np.sum(loss))

            # 
            self.w = self.w - self.learning_rate * self.dot(X.T, loss)
            # self.w = self.w - self.learning_rate * X.T * loss

        return loss_history

    def predict(self, x_test):
    
        labels = self.dot(x_test, self.w)
        labels = self.sigmoid(labels)
        labels[labels>=0.5] = 1
        labels[labels<0.5] = 0
        return labels

    def dot(self, A, B):
        try:
            m, n = A.shape
            x, y = B.shape
        except:
            m = 1
            n = len(A)
            x, y = B.shape

        res = np.zeros((m, y))
        for i in range(m):
            val = 0.0
            for j in range(y):
                for k in range(n):
                    val += A[i][k] * B[k][j]
                res[i][j] = val

        return res
