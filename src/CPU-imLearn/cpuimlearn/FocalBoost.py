# Reference :
#
# Name : FocalBoost
#
# Purpose : FocalBoost is an classification algorithm for multi-imbalanced data, which applies the focal loss in Boosting method.
#
# This file is a part of GPU-imLearn software, A software for imbalance data classification based on GPU.
# 
# GPU-imLearn software is distributed in the hope that it will be useful,but WITHOUT ANY WARRANTY; without even the implied warranty of \n
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class FocalBoost:

    def __init__(self):

        self.clf = None
        print('FocalBoost.FocalBoost')

    def softmax(self, x):
        """softmax"""
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))

        return x

    # focal loss in this work.
    # cls : 1 to n
    def focal(self, nrows, ncls, pre, y):

        isim = np.zeros(nrows)
        for x in self.imcls:
            isim[y == x] = 1

        logit = self.softmax(pre)
        one_hot_key = np.zeros((nrows, ncls))
        for i in range(nrows):
            col = int(y[i]-1)
            one_hot_key[i][col] = 1

        pt = (one_hot_key * logit).sum(1)
        logpt = np.log(pt + 0.000001)
        logpt2 = np.log(1-pt)

        loss1 = -0.25 * np.power((1-(pt*isim)), 2) * logpt
        loss0 = -1 * np.power((pt*(1-isim)), 2) * logpt2

        loss = loss0 + loss1

        return loss.mean()
    
    # weak classification in iteration.
    def create_classifer(self, index=0):
        # return DecisionTreeClassifier()
        return RandomForestClassifier(n_estimators=50, max_depth=20)

    def resample(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        result_arr = np.searchsorted(t, np.random.rand(len(weights))*s)
        return result_arr

    def train(self, x, y):
        y = np.array(y, dtype='int64')
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

            weak_classifier = self.create_classifer()
            weak_classifier.fit(resampled_entries_x, resampled_entries_y, sample_weight=weights)

            # training and calculate the rate of error
            classifications = weak_classifier.predict(x)

            pre = weak_classifier.predict_proba(x)
            error = self.focal(num_rows, ncls, pre, y) * np.array(weights)
            alpha = np.sum(error)

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

            cur_max_value = -1000
            max_key = ''
            for key in result_map:

                if result_map[key] > cur_max_value:

                    cur_max_value = result_map[key]
                    max_key = key

            res.append(float(max_key))
        return np.asarray(res)

    # imcls is an array which contains the minority classes.
    def fit(self, x_train, y_train, imcls=[], ismulti=1, n_rounds=20):
        # ncls = len(np.unique(y_train))
        self.imcls = np.array(imcls)
        self.ismulti = ismulti
        self.num_rounds = n_rounds
        self.train(x_train, y_train.ravel())

    def predict(self, x_test):
        pre_label = (self.predicted(x_test)).reshape(-1, 1)
        pre_label = np.squeeze(pre_label)

        return pre_label
