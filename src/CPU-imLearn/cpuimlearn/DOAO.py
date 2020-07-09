# Reference :
#
# Name : DOVO
#
# Purpose : DOVO is an classification algorithm for multi-imbalanced data.
# 
# This file is a part of GPU-imLearn software, A software for imbalance data classification based on GPU.
# 
# GPU-imLearn software is distributed in the hope that it will be useful,but WITHOUT ANY WARRANTY; without even the implied warranty of \n
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.stats import mode
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


class DOAO:

    def __init__(self):
        print('DOAO.DOAO')

    def run(self, x_train, x_test, y_train, y_test):

        start_time = time.time()

        # kfold, default value is 5
        kfold = 5

        train = np.append(x_train, y_train, axis=1)
        cls = np.unique(y_train)
        ncls = len(cls)

        C, D, L = [[] for c in range(2)], [], []
        # C 是验证过后， 对每组 D 表现最好的算法和K的记录
        # D 是只有两个标签值的每组数据
        # L 是数据本来的标签 i，j ， 即 nlabel 记录的标签值
        for i in range(ncls-1):
            for j in range(i+1, ncls):
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

                D.append(data_ij)
                C[0].append(fun_best)
                C[1].append(bestk)
                L.append(nlabel)

        # predict
        pre_label = self.fun_predict(x_test, y_test, C, D, L)
        pre_label = np.squeeze(pre_label)
        acc, f1_score, g_mean = fac.get_acc(y_test, pre_label)

        end_time = time.time()
        cost = end_time-start_time

        return pre_label, acc, f1_score, g_mean, cost

    def get_classifier(self, traindata, kf):

        x_tr, x_te, y_tr, y_te = fac.to_kfold(traindata, kf)
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


        acc = np.array(acc)
        acc_mean = acc.mean(axis=0)
        # fun_best = np.where(acc_mean == max(acc_mean))
        fun_best = np.argmax(acc_mean)

        return fun_best, bestK

    def fun_predict(self, x_te, y_te, C, D, L):
        num = len(D)
        cf = C[0]
        ck = C[1]
        allpre = np.zeros((len(y_te), num))
        for i in range(num):
            train = D[i]
            traindata = train[:, 0:-1]
            trainlabel = train[:, -1]

            if cf[i] == 0:
                # svm
                clf_svm = SVC(gamma='scale')
                clf_svm.fit(traindata, trainlabel)
                label_svm = clf_svm.predict(x_te)
                allpre[:, i] = label_svm
            elif cf[i] == 1:
                # knn
                clf_knn = KNeighborsClassifier(n_neighbors=ck[i])
                clf_knn.fit(traindata, trainlabel)
                label_knn = clf_knn.predict(x_te)
                allpre[:, i] = label_knn
            elif cf[i] == 2:
                # LR
                clf_lr = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
                clf_lr.fit(traindata, trainlabel)
                label_LR = clf_lr.predict(x_te)
                allpre[:, i] = label_LR
            elif cf[i] == 3:
                # C4.5
                clf_c45 = DecisionTreeClassifier(criterion='entropy')
                clf_c45.fit(traindata, trainlabel)
                label_c45 = clf_c45.predict(x_te)
                allpre[:, i] = label_c45
            elif cf[i] == 4:
                # AdaBoost
                clf_ada = AdaBoostClassifier(n_estimators=50)
                clf_ada.fit(traindata, trainlabel)
                label_ada = clf_ada.predict(x_te)
                allpre[:, i] = label_ada
            elif cf[i] == 5:
                # RF
                clf_rf = RandomForestClassifier(n_estimators=50)
                clf_rf.fit(traindata, trainlabel)
                label_rf = clf_rf.predict(x_te)
                allpre[:, i] = label_rf
            elif cf[i] == 6:
                # CART
                clf_cart = DecisionTreeClassifier(criterion='gini')
                clf_cart.fit(traindata, trainlabel)
                label_cart = clf_cart.predict(x_te)
                allpre[:, i] = label_cart
            else:
                print('error !!!! DOAO.fun_predict')

            label = L[i]
            for j in range(len(y_te)):
                allpre[j,i] = label[0] if allpre[j,i] == 0 else label[1]

        pre = mode(allpre, axis=1)[0]

        return pre
