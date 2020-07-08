import os
import time
import scipy.io as sio
import glob
import numpy as np

from gpuimlearn import FocalBoost

def norm(matrix):
    ran = np.max(matrix) - np.min(matrix)
    return ((matrix - np.min(matrix))/ran)

def main():
    # 当前工作目录
    dir_path = os.getcwd()
    result_path = os.path.join(dir_path, 'result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    # 获取data目录下的mat数据集文件
    for data_path in glob.glob(r'%s/data/*.mat' % dir_path):
        fname = os.path.splitext(os.path.basename(data_path))[0]
        load_data = sio.loadmat(data_path)['data'][0]

        start_time = time.time()
        # 训练、预测
        # K交叉验证(受数据集限制 K=5)
        global x_train, y_train, x_test, y_test
        for i in range(load_data.size):
            x_train = load_data[i]['train']
            x_test = load_data[i]['test']
            y_train = load_data[i]['trainlabel']
            y_test = load_data[i]['testlabel']
            x_train, x_test = norm(x_train), norm(x_test)

            clf = FocalBoost.FocalBoost()
            clf.fit(x_train, y_train.ravel())
            pre = clf.predict(x_test)
            print(pre)

        end_time = time.time()
        print('\nSet  : %s\nCost : %s s\nDone...........................\n' % (fname, (end_time - start_time)))


if __name__ == "__main__":
    main()