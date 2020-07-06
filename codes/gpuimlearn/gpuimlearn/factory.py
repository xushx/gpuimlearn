# @Reference
# @Time : ${DATE} ${TIME}
# @Author : henu_bigdata
# @File : factory.py

import numpy as np
from sklearn.model_selection import KFold


class FAC:

    def __init__(self):
        # print('....factory.FAC')
        pass

    def to_kfold(self, traindata, kfold):
        # data to kfold
        x_tr, x_te, y_tr, y_te = [], [], [], []
        kf_data = KFold(n_splits=kfold, shuffle=True)

        for tr, te in kf_data.split(traindata):
            x_tr.append(traindata[tr, 0:-1])
            x_te.append(traindata[te, 0:-1])
            y_tr.append(traindata[tr, -1])
            y_te.append(traindata[te, -1])

        x_tr = np.array(x_tr)
        x_te = np.array(x_te)
        y_tr = np.array(y_tr)
        y_te = np.array(y_te)

        return x_tr, x_te, y_tr, y_te

    def get_acc(self, ac_label, pre_label):
        ac_label = np.squeeze(ac_label)
        pre_label = np.squeeze(pre_label)
        tp, tn, fn, fp, corr, err = 0, 0, 0, 0, 0, 0
        for i, j in zip(ac_label, pre_label):
            if i == j:
                # corr += 1
                if i == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                # err += 1
                if i == 1:
                    fn += 1
                else:
                    fp += 1
        acc = (tp+tn)/(tp+tn+fp+fn)
        recall = tp/(tp+fn+0.0001)
        precision = tp/(tp+fp+0.0001)
        specificity = tn/(tn+fp+0.0001)
        f1_score = (2*precision*recall)/(precision+recall+0.0001)
        g_mean = pow(recall*specificity, 0.5)

        return acc, f1_score, g_mean

    def get_type(self, data):

        num_all = np.size(data)
        num0 = len(np.where(data == 0))
        rate = num0 / num_all
        if rate >= 0.3:
            type = 'sparse'
        else:
            type = 'dense'

        return type
        # return 'sparse'
