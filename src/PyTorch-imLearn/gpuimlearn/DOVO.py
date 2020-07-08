# @Reference
# @Time : ${DATE} ${TIME}
# @File : DOVO.py
# @Description : a classifier for imbalance data based on GPU

import numpy as np
from scipy.stats import mode

from thundergbm import TGBMClassifier
from thundersvm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from gpuimlearn.func_base_gpu.KNN_GPU import KNN_GPU
from gpuimlearn.func_base_gpu.LR_GPU import LR_GPU

from gpuimlearn.factory import FAC
fac = FAC()


class DOVO(object):

    def __init__(self):
        print('DOAO.DOAO')

    def fit(self, x_train , y_train):
        kfold = 5

        train = np.append(x_train, y_train, axis=1)
        cls = np.unique(y_train)
        ncls = len(cls)

        self.C, self.D, self.L = [[] for c in range(2)], [], []

        for i in range(ncls - 1):
            for j in range(i + 1, ncls):
                id_i = np.where(y_train == cls[i])[0]
                id_j = np.where(y_train == cls[j])[0]

                # data_ij is the set of data instances whose class labels are either i or j
                data_ij = np.append(train[id_i, :], train[id_j, :], axis=0)
                data_ij_label = data_ij[:, -1]
                nlabel = np.unique(data_ij_label)

                # transform class labels from (i ,j) to (0 , 1)
                aux_arr = data_ij_label == nlabel[0]
                data_ij_label[aux_arr] = 0
                data_ij_label[np.logical_not(aux_arr)] = 1

                data_ij[:, -1] = data_ij_label

                # classifier_best, k_best = self.get_classifier(data_ij, kfold)
                fun_best, bestk = self.get_classifier(data_ij, kfold)

                self.D.append(data_ij)
                self.C[0].append(fun_best)
                self.C[1].append(bestk)
                self.L.append(nlabel)

    def predict(self, x_test):

        pre_label = self.fun_predict(x_test, self.C, self.D, self.L)
        pre_label = np.squeeze(pre_label)

        return pre_label


    def get_classifier(self, traindata, kf):

        x_tr, x_te, y_tr, y_te = fac.to_kfold(traindata, kf)
        acc_max, bestK, acc = 0, 0, [[] for a in range(kf)]
        
        for i in range(kf):
        
            # print('DOAO round', i, 'begin')
            # svm 00
            print('test00')
            clf_svm = SVC()
            clf_svm.fit(x_tr[i], y_tr[i].ravel())
            label_svm = clf_svm.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_svm)[0])

            # KNN 01
            print('test01')
            acc_k = []
            aux_k = [3, 5, 7]
            # for k in range(3, 12, 2):
            for k in aux_k:
                clf_knn = KNN_GPU(k=k)
                clf_knn.fit(x_tr[i], y_tr[i])
                label_knn = clf_knn.predict(x_te[i])
                acc_k.append(fac.get_acc(y_te[i], label_knn)[0])
            acc[i].append(max(acc_k))
            bestK = aux_k[acc_k.index(max(acc_k))]
            
            # LR 02
            print('test02')
            clf_lr = LogisticRegression()
            clf_lr.fit(x_tr[i], y_tr[i])
            label_LR = clf_lr.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_LR)[0])
            # # LR 02
            # clf_lr = LR_GPU()
            # clf_lr.fit(x_tr[i], y_tr[i])
            # label_LR = clf_lr.predicted(x_te[i])
            # acc[i].append(fac.get_acc(y_te[i], label_LR)[0])

            # XgBoost 03
            print('test03')
            clf_xgb = DecisionTreeClassifier()
            clf_xgb.fit(x_tr[i], y_tr[i])
            label_xgb = clf_xgb.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_xgb)[0])

            # RF 04
            print('test04')
            
            clf_rf = TGBMClassifier()
            clf_rf.fit(x_tr[i], y_tr[i])
            label_rf = clf_rf.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_rf)[0])
            
            print('DOAO round', i, 'end')

        acc = np.array(acc)
        acc_mean = acc.mean(axis=0)
        
        # fun_best = np.where(acc_mean == max(acc_mean))
        fun_best = np.argmax(acc_mean)

        return fun_best, bestK

    def fun_predict(self, x_te, C, D, L):
        print('func_predict')

        num = len(D)
        cf = C[0]
        ck = C[1]

        allpre = np.zeros((len(x_te), num))
        for i in range(num):
            train = D[i]
            traindata = train[:, 0:-1]
            trainlabel = train[:, -1]

            if cf[i] == 0:
                # svm
                print('SVM predict')
                clf_svm = SVC()
                clf_svm.fit(traindata, trainlabel.ravel())
                label_svm = clf_svm.predict(x_te)
                allpre[:, i] = label_svm
            elif cf[i] == 1:
                # knn
                clf_knn = KNN_GPU(k=ck[i])
                clf_knn.fit(traindata, trainlabel)
                label_knn = clf_knn.predict(x_te)
                allpre[:, i] = label_knn
            elif cf[i] == 2:
                # LR
                print('LR predict')
                clf_lr = LogisticRegression()
                clf_lr.fit(traindata, trainlabel.ravel())
                label_LR = clf_lr.predict(x_te)
                allpre[:, i] = label_LR
                # # LR
                # clf_lr = LR_GPU()
                # clf_lr.fit(traindata, trainlabel)
                # label_LR = clf_lr.predicted(x_te)
                # allpre[:, i] = label_LR
            elif cf[i] == 3:
                # CART
                print('CART predict')
                clf_xgb = DecisionTreeClassifier()
                clf_xgb.fit(traindata, trainlabel)
                label_xgb = clf_xgb.predict(x_te)
                allpre[:, i] = label_xgb
            elif cf[i] == 4:
                # Rf
                print('RF predict')
                clf_rf = TGBMClassifier()
                clf_rf.fit(traindata, trainlabel.ravel())
                label_rf = clf_rf.predict(x_te)
                allpre[:, i] = label_rf
            else:
                print('error !!!! DOAO.fun_predict')


            label = L[i]
            for j in range(len(x_te)):
                allpre[j, i] = label[0] if allpre[j, i] == 0 else label[1]

        print('predict end for')
        pre = mode(allpre, axis=1)[0]
        return pre
