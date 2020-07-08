import numpy as np
import time

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from factory import FAC
fac = FAC()


class DECOC:

    def __init__(self):
        print('DECOC.DECOC')

    def run(self, x_train, x_test, y_train, y_test):

        type, withw = fac.get_type(x_train), 1

        start_time = time.time()

        pre_label = []
        code, ft, labels, D = self.funClassifierDECOC(x_train, y_train, type)
        W = self.funcwEDOVO(x_train, y_train, code, ft, labels, D)
        if withw == 0:
            W[0:len(ft)+1] = 1

        pre = self.funcPreTestEDOVO(x_test, code, ft, W, D)
        for i in range(len(pre)):
            pre_label.append(labels[pre[i]])
        pre_label = np.squeeze(pre_label)
        acc, f1_score, g_mean = fac.get_acc(y_test, pre_label)

        end_time = time.time()
        cost = end_time - start_time

        return pre_label, acc, f1_score, g_mean, cost

    def funClassifierDECOC(self, x_train, y_train, type):

        labels = np.unique(y_train)
        nc = len(labels)
        code1 = self.funECOCim(nc, type)
        code1 = np.asarray(code1)
        train, leng = [[] for c in range(nc)], []
        for i in range(nc):
            id_i = np.where(y_train == labels[i])[0]
            train[i] = x_train[id_i, :]
            leng.append(len(y_train[id_i]))

        numberl = np.size(code1, 1)

        ft, D = [[] for c in range(2)], []
        for t in range(numberl):
            Dt, DtLabel, flagDt, numAP, numAN, numP, numN = np.asarray([]), [], 0, 0, 0, 0, 0
            for i in range(nc):
                if Dt.shape[0] == 0:
                    Dt = train[i]
                else:
                    Dt = np.append(Dt, train[i], axis=0)
                DtLabel[flagDt : flagDt+leng[i]-1] = [code1[i, t]] * leng[i]
                flagDt += leng[i]

                if code1[i, t] == 1:
                    numAP += 1
                    numP += leng[i]
                elif code1[i, t] == -1:
                    numAN += 1
                    numN += leng[i]

            ct, flagct = [], 0
            for j in range(nc):
                if code1[j, t] == 1:
                    cti = max(numP, numN) / np.dot(numAP, leng[j])
                    ct[flagct : flagct + leng[j] - 1] = [cti] * leng[j]
                    flagct += leng[j]
                elif code1[j, t] == -1:
                    cti = max(numP, numN) / np.dot(numAN, leng[j])
                    ct[flagct: flagct + leng[j] - 1] = [cti] * leng[j]
                    flagct += leng[j]

            fun_best, bestk = self.get_classifier(np.column_stack((Dt, np.asarray(DtLabel).T)), 5)

            ft[0].append(fun_best)
            ft[1].append(bestk)

            D.append(np.column_stack((Dt, np.asarray(DtLabel).T)))

        ft, D = np.asarray(ft).T, np.asarray(D)
        return code1, ft, labels, D

    def funcwEDOVO(self, x_train, y_train, code, ft, labels, D):

        W = []
        numset = len(y_train)
        # ????
        W[0:len(ft)] = [np.sqrt(1/len(ft))]*len(ft)
        W = np.asarray(W)

        fX = self.funcPreEDOVO(x_train, y_train, ft, D)
        fX = np.asarray(fX)
        fX[np.where(fX == 0)] = -1

        ny, gama = [], []
        for i in range(len(labels)):
            ny.append(len(np.where(y_train == labels[i])))
        for i in range(len(labels)):
            gama.append(max(ny)/ny[i])

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

    def funcPreTestEDOVO(self, x_test, code, ft, W, D):

        pre_label = []
        num_set = np.asarray(x_test).shape[0]
        ncls = np.asarray(code).shape[0]
        y_test = np.zeros(num_set)
        # ????
        y_test[1] = 1

        fX = self.funcPreEDOVO(x_test, y_test, ft, D)
        fX[np.where(fX == 0)] = -1

        for i in range(num_set):
            ftx = fX[i]
            yall = []
            for j in range(ncls):
                btr = []
                for k in range(len(ftx)):
                    btr.append((1 - np.dot(ftx[k], code[j, k]))/2)
                br = np.asarray(btr).T
                yall.append(np.dot(W, br))
            minindx = yall.index(min(yall))
            pre_label.append(minindx)

        pre_label = np.asarray(pre_label)

        return pre_label

    def funcPreEDOVO(self, x_test, y_test, C, D):

        numC = np.asarray(C).shape[0]
        num_set = len(y_test)
        allpre = np.zeros([num_set, numC])

        for i in range(numC):
            train = D[i]
            traindata = train[:, 0:-1]
            trainlabel = (train[:, -1]).astype('int')
            if C[i, 0] == 0:
                # svm
                clf_svm = SVC(gamma='scale')
                clf_svm.fit(traindata, trainlabel)
                label_svm = clf_svm.predict(x_test)
                allpre[:, i] = label_svm
            elif C[i, 0] == 1:
                # knn
                clf_knn = KNeighborsClassifier(n_neighbors=C[i][1])
                clf_knn.fit(traindata, trainlabel)
                label_knn = clf_knn.predict(x_test)
                allpre[:, i] = label_knn
            elif C[i, 0] == 2:
                # LR
                clf_lr = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
                clf_lr.fit(traindata, trainlabel)
                label_LR = clf_lr.predict(x_test)
                allpre[:, i] = label_LR
            elif C[i, 0] == 3:
                # C4.5
                clf_c45 = DecisionTreeClassifier(criterion='entropy')
                clf_c45.fit(traindata, trainlabel)
                label_c45 = clf_c45.predict(x_test)
                allpre[:, i] = label_c45
            elif C[i, 0] == 4:
                # AdaBoost
                clf_ada = AdaBoostClassifier(n_estimators=50)
                clf_ada.fit(traindata, trainlabel)
                label_ada = clf_ada.predict(x_test)
                allpre[:, i] = label_ada
            elif C[i, 0] == 5:
                # RF
                clf_rf = RandomForestClassifier(n_estimators=50)
                clf_rf.fit(traindata, trainlabel)
                label_rf = clf_rf.predict(x_test)
                allpre[:, i] = label_rf
            elif C[i, 0] == 6:
                # CART
                clf_cart = DecisionTreeClassifier(criterion='gini')
                clf_cart.fit(traindata, trainlabel)
                label_cart = clf_cart.predict(x_test)
                allpre[:, i] = label_cart
            elif C[i, 0] == 7:
                # MLP
                clf_mlp = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(50, 50), max_iter=1000,
                                        learning_rate='adaptive')
                clf_mlp.fit(traindata, trainlabel)
                label_mlp = clf_mlp.predict(x_test)
                allpre[:, i] = label_mlp
            else:
                print('error !!!! DECOC.funcPreEDOVO')

        return allpre

    def funECOCim(self, ncls, type):

        code = []
        if type == 'OVA':
            code = self.funOVAim(ncls)
        elif type == 'OVO':
            code = self.funOVOim(ncls)
        elif type == 'dense':
            N_dichotomizers = int(min(pow(2, ncls - 1) - 1, np.floor(10 * np.log2(ncls))))
            zero_prob = 0.0
            code = self.pseudoRanddomCoding(ncls, N_dichotomizers, zero_prob)
        elif type == 'sparse':
            N_dichotomizers = int(min((pow(3, ncls) - 2 * pow(2, ncls) + 1) / 2, np.floor(15 * np.log2(ncls))))
            zero_prob = 0.5
            code = self.pseudoRanddomCoding(ncls, N_dichotomizers, zero_prob)
        else:
            print('Error....funECOCim')

        return code

    def funOVAim(self, ncls):

        code = np.ones([ncls, ncls])
        code = np.asarray(-code)
        i = np.arange(ncls)
        code[i, i] = 1

        return code

    def funOVOim(self, ncls):

        cnum = ncls*(ncls-1)/2
        code = np.zeros([ncls, cnum])
        flag = 0
        for i in range(ncls-1):
            for j in range(i, ncls):
                code[i, flag] = 1
                code[j, flag] = -1
                flag += 1

        return code

    def pseudoRanddomCoding(self, ncls, col, zero_prob):

        classes = np.arange(ncls)
        cls_len = len(classes)
        iterations = cls_len < 5 and 1000 or 5000
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

    # def get_classifier(self, train, kf):
    #
    #     x_tr, x_te, y_tr, y_te = fac.to_kfold(train, kf)
    #     acc_max, bestK, acc = 0, 0, [[] for a in range(kf)]
    #
    #     for i in range(kf):
    #         # svm 00
    #         clf_svm = SVC(gamma='scale')
    #         clf_svm.fit(x_tr[i], y_tr[i])
    #         label_svm = clf_svm.predict(x_te[i])
    #         acc[i].append(fac.get_acc(y_te[i], label_svm)[0])
    #
    #         # KNN 01
    #         acc_k = []
    #         aux_k = [3, 5, 7, 9, 11]
    #         # for k in range(3, 12, 2):
    #         for k in aux_k:
    #             clf_knn = KNeighborsClassifier(n_neighbors=k)
    #             clf_knn.fit(x_tr[i], y_tr[i])
    #             label_knn = clf_knn.predict(x_te[i])
    #             acc_k.append(fac.get_acc(y_te[i], label_knn)[0])
    #         acc[i].append(max(acc_k))
    #         bestK = aux_k[acc_k.index(max(acc_k))]
    #
    #         # LR 02
    #         clf_lr = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
    #         # {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’} default = 'lbfgs'
    #         # ‘newton-cg’是牛顿家族中的共轭梯度法，‘lbfgs’是一种拟牛顿法，‘newton-cg’是牛顿家族中的共轭梯度法，‘lbfgs’是一种拟牛顿法，
    #         # ‘sag’则是随机平均梯度下降法，‘saga’是随机优化算法，‘liblinear’‘sag’则是随机平均梯度下降法，‘saga’是随机优化算法，‘liblinear’
    #         # 是坐标轴下降法
    #         clf_lr.fit(x_tr[i], y_tr[i])
    #         label_LR = clf_lr.predict(x_te[i])
    #         acc[i].append(fac.get_acc(y_te[i], label_LR)[0])
    #
    #         # C4.5 03
    #         clf_c45 = DecisionTreeClassifier(criterion='entropy')
    #         clf_c45.fit(x_tr[i], y_tr[i])
    #         label_c45 = clf_c45.predict(x_te[i])
    #         acc[i].append(fac.get_acc(y_te[i], label_c45)[0])
    #
    #         # AdaBoost 04
    #         clf_ada = AdaBoostClassifier(n_estimators=20)
    #         clf_ada.fit(x_tr[i], y_tr[i])
    #         label_ada = clf_ada.predict(x_te[i])
    #         acc[i].append(fac.get_acc(y_te[i], label_ada)[0])
    #
    #         # RF 05
    #         clf_rf = RandomForestClassifier(n_estimators=50)
    #         clf_rf.fit(x_tr[i], y_tr[i])
    #         label_rf = clf_rf.predict(x_te[i])
    #         acc[i].append(fac.get_acc(y_te[i], label_rf)[0])
    #
    #         # CART 06
    #         clf_cart = DecisionTreeClassifier(criterion='gini')
    #         clf_cart.fit(x_tr[i], y_tr[i])
    #         label_cart = clf_cart.predict(x_te[i])
    #         acc[i].append(fac.get_acc(y_te[i], label_cart)[0])
    #
    #         # # MLP 07
    #         # clf_mlp = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(50, 50), max_iter=1000, learning_rate='adaptive')
    #         # clf_mlp.fit(x_tr[i], y_tr[i])
    #         # label_mlp = clf_mlp.predict(x_te[i])
    #         # acc[i].append(fac.get_acc(y_te[i], label_mlp)[0])
    #
    #     acc = np.array(acc)
    #     acc_mean = acc.mean(axis=0)
    #     # fun_best = np.where(acc_mean == max(acc_mean))
    #     fun_best = np.argmax(acc_mean)
    #
    #     return fun_best, bestK

    def get_classifier(self, train, kf):

        x_tr, x_te, y_tr, y_te = fac.to_kfold(train, kf)
        acc_max, bestK, acc = 0, 0, [[] for a in range(kf)]

        for i in range(kf):
            # svm 00
            clf_svm = SVC(gamma='scale')
            clf_svm.fit(x_tr[i], y_tr[i])
            label_svm = clf_svm.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_svm)[0])

            # KNN 01
            acc_k = []
            aux_k = [3, 5, 7, 9, 11]
            # for k in range(3, 12, 2):
            for k in aux_k:
                clf_knn = KNeighborsClassifier(n_neighbors=k)
                clf_knn.fit(x_tr[i], y_tr[i])
                label_knn = clf_knn.predict(x_te[i])
                acc_k.append(fac.get_acc(y_te[i], label_knn)[0])
            acc[i].append(max(acc_k))
            bestK = aux_k[acc_k.index(max(acc_k))]

            # LR 02
            clf_lr = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
            # {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’} default = 'lbfgs'
            # ‘newton-cg’是牛顿家族中的共轭梯度法，‘lbfgs’是一种拟牛顿法，‘newton-cg’是牛顿家族中的共轭梯度法，‘lbfgs’是一种拟牛顿法，
            # ‘sag’则是随机平均梯度下降法，‘saga’是随机优化算法，‘liblinear’‘sag’则是随机平均梯度下降法，‘saga’是随机优化算法，‘liblinear’
            # 是坐标轴下降法
            clf_lr.fit(x_tr[i], y_tr[i])
            label_LR = clf_lr.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_LR)[0])

            # C4.5 03
            clf_c45 = DecisionTreeClassifier(criterion='entropy')
            clf_c45.fit(x_tr[i], y_tr[i])
            label_c45 = clf_c45.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_c45)[0])

            # AdaBoost 04
            clf_ada = AdaBoostClassifier(n_estimators=20)
            clf_ada.fit(x_tr[i], y_tr[i])
            label_ada = clf_ada.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_ada)[0])

            # RF 05
            clf_rf = RandomForestClassifier(n_estimators=50)
            clf_rf.fit(x_tr[i], y_tr[i])
            label_rf = clf_rf.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_rf)[0])

            # CART 06
            clf_cart = DecisionTreeClassifier(criterion='gini')
            clf_cart.fit(x_tr[i], y_tr[i])
            label_cart = clf_cart.predict(x_te[i])
            acc[i].append(fac.get_acc(y_te[i], label_cart)[0])

            # # MLP 07
            # clf_mlp = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(50, 50), max_iter=1000, learning_rate='adaptive')
            # clf_mlp.fit(x_tr[i], y_tr[i])
            # label_mlp = clf_mlp.predict(x_te[i])
            # acc[i].append(fac.get_acc(y_te[i], label_mlp)[0])

        acc = np.array(acc)
        acc_mean = acc.mean(axis=0)
        # fun_best = np.where(acc_mean == max(acc_mean))
        fun_best = np.argmax(acc_mean)

        return fun_best, bestK
