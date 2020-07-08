# @Reference
# @Time : ${DATE} ${TIME}
# @File : imECOC.py
# @Description : a classifier for imbalance data based on GPU

import numpy as np
import time

from thundersvm import SVC


class imECOC(object):


    def fit(self, x_train, y_train):
        type, withw = 'sparse', 1
        self.code, self.ft, self.labels = self.funClassifier(x_train, y_train, type)
        self.W = self.funcW(x_train, y_train, self.code, self.ft, self.labels)
        if withw == 0:
            self.W[0:len(self.ft)] = [1] * len(self.ft)

    def predict(self, x_test):
        pre_label = []
        pre = self.funcPre(x_test, self.code, self.ft, self.W)

        for i in range(len(pre)):
            pre_label.append(self.labels[pre[i]])
        pre_label = np.squeeze(pre_label)

        return pre_label

    def funClassifier(self, x_train, y_train, type):
        labels = np.unique(y_train)
        nc = len(labels)
        code = self.funECOCim(nc, type)

        train, numn = [], []
        for i in range(nc):
            idi = np.where(y_train == labels[i])[0]
            train.append(x_train[idi])
            numn.append(len(idi))

        num1 = np.size(code, 1)
        ft = []
        for t in range(num1):
            Dt, DtLabel, flagDt, numAp, numAn, numNp, numNn = np.asarray([]), [], 0, 0, 0, 0, 0
            for i in range(nc):
                if code[i, t] == 1:
                    if Dt.shape[0] == 0:
                        Dt = train[i]
                    else:
                        Dt = np.append(Dt, train[i], axis=0)
                    DtLabel[flagDt: flagDt + numn[i]] = [1] * numn[i]
                    flagDt += numn[i]
                    numAp += 1
                    numNp += numn[i]
                elif code[i, t] == -1:
                    if Dt.shape[0] == 0:
                        Dt = train[i]
                    else:
                        Dt = np.append(Dt, train[i], axis=0)
                    DtLabel[flagDt: flagDt + numn[i]] = [0] * numn[i]
                    flagDt += numn[i]
                    numAn += 1
                    numNn += numn[i]

            clf_svc = SVC()
            clf_svc.fit(np.array(Dt), np.array(DtLabel).ravel())
            ft.append(clf_svc)

        return code, ft, labels

    def funECOCim(self, ncls, type):

        if type == '':
            type = 'sparse'

        type = 'sparse'
        code = []
        if type == 'OVA':
            code = self.funOVA(ncls)
        elif type == 'dense':
            N_dichotomizers = int(min(pow(2, ncls - 1) - 1, np.floor(10 * np.log2(ncls))))
            zero_prob = 0.0
            code = self.pseudoRanddomCoding(ncls, N_dichotomizers, zero_prob)
        elif type == 'sparse':
            N_dichotomizers = int(min((pow(3, ncls) - 2 * pow(2, ncls) + 1) / 2, np.floor(15 * np.log2(ncls))))
            print('Code_Matrix : (%s, %s)\n' % (ncls, N_dichotomizers))
            zero_prob = 0.5
            code = self.pseudoRanddomCoding(ncls, N_dichotomizers, zero_prob)
        else:
            print('Error....funECOCim')

        return code

    def funOVA(self, ncls):

        code = np.ones([ncls, ncls])
        code = np.asarray(-code)
        i = np.arange(ncls)
        code[i, i] = 1

        return code

    def pseudoRanddomCoding(self, ncls, col, zero_prob):

        classes = np.arange(ncls)
        cls_len = len(classes)
        iterations = cls_len < 5 and 50 or 100
        max_min_pair_distance = 0
        ECOC_Matrix = np.zeros([cls_len, col])
        for iter_count in range(iterations):
            ECOCs = np.zeros([cls_len, col])
            for z in range(col):
                satistified_condition = 0
                while satistified_condition == 0:
                    if zero_prob == 0:
                        tmp = np.floor(2 * np.random.rand(cls_len))
                        tmp[np.where(tmp == 0)] = -1
                        ECOCs[:, z] = tmp
                    elif zero_prob == 0.5:
                        ECOCs[:, z]=np.round(2 * np.random.rand(cls_len) - 1)
                    elif zero_prob == 1/3:
                        ECOCs[:, z]=np.floor(3 * np.random.rand(cls_len) - 1)

                    satistified_condition = 1
                    if (sum(ECOCs[:, z] == 1) == 0 or sum(ECOCs[:,z] == -1) == 0):
                        satistified_condition = 0
                    else:
                        for c in range(z-1):
                            if (ECOCs[:, z] == ECOCs[:, c]).all() or (ECOCs[:, z] == -ECOCs[:, c]).all():
                                satistified_condition = 0
            counter = 0
            min_pair_distan = 1000

            for j in range(cls_len-1):
                for k in range(j+1, cls_len):
                    # pair_distance = 0.5 * sum((1 - (ECOCs[j, :] * ECOCs[k, :])) * abs(ECOCs[j, :] * ECOCs[k, :]))
                    pair_distance = 0.5 * sum((1 - (ECOCs[j] * ECOCs[k])) * abs(ECOCs[j] * ECOCs[k]))
                    counter += 1
                    # min_pair_distan = pair_distance < min_pair_distan and pair_distance
                    if pair_distance < min_pair_distan:
                        min_pair_distan = pair_distance

            if min_pair_distan > max_min_pair_distance:
                max_min_pair_distance = min_pair_distan
                ECOC_Matrix = ECOCs

        return ECOC_Matrix

    def funcW(self, x_train, y_train, code, ft, labels):
        W = []
        numset = len(y_train)
        W[0:len(ft)] = [np.sqrt(1 / len(ft))] * len(ft)
        W = np.asarray(W)

        fX = []
        for i in ft:
            label_ft = np.asarray(i.predict(x_train))
            fX.append(label_ft)
        fX = np.asarray(fX).T

        ny, gama = [], []
        for i in range(len(labels)):
            ny.append(len(np.where(y_train == labels[i])))
        for i in range(len(labels)):
            gama.append(max(ny) / ny[i])

        for i in range(numset):
            ftx = fX[i]
            indx = int(np.where(labels == y_train[i])[0])
            yi = code[indx]
            # print(fX, code, indx, ftx, yi)
            for j in range(len(ftx)):
                if ftx[j] != yi[j]:
                    btyt = (1 - np.dot(ftx[j], yi[j]))/2
                    W[j] += np.dot(gama[indx], btyt)

        W = np.sqrt(W / sum(W))

        return W

    def funcPre(self, x_test, code, ft, W):
        pre_label = []
        num_set, ncls = len(x_test), np.size(code, 0)

        fX = []
        for i in ft:
            label_ft = np.asarray(i.predict(x_test))
            fX.append(label_ft)
        fX = np.asarray(fX).T

        for i in range(num_set):
            ftx = fX[i]
            yall = []
            for j in range(ncls):
                btr = []
                for k in range(len(ftx)):
                    btr.append((1 - np.dot(ftx[k], code[j, k])) / 2)
                br = np.asarray(btr).T
                yall.append(np.dot(W, br))
            minindx = yall.index(min(yall))
            pre_label.append(minindx)

        return pre_label
