# GPU-imLearn
A software for imbalance data classification based on GPU. <br>
It can provide high speed when you classify the multi-class imbalanced data. <br>

## Copyright
This work was designed by Prof. Chongsheng Zhang (chongsheng.zhang@yahoo.com), and implemented by Mr. Shixin Xu (xusxmail@qq.com), who is a master student of Henan University under the supervision of Prof. Zhang. <br>

This software is free for academic use only. For commercial companies, they should first ask the permission from both authors above. <br>

## Document Description
* codes : the code work. there is a sample supported.
* data : 2 sample datasets, which are '.mat' format.
* docs : some related papers, and user manual.

## Software Contents
There are 3 main parts in this work, you can see them in the codes folder. <br>
* cpu <br>
6 multi-class imbalanced classification algorithms based on CPU
* gpuimlearn -recommend- <br>
6 multi-class imbalanced classification algorithms based on GPU, base classifications are implemented by CUDA.
* pytorch <br>
6 multi-class imbalanced classification algorithms based on GPU, base classifications are implemented by Pytorch.

the 6 multi-class imbalanced classification algorithms are :
1. FocalBoost 
2. DECOC 
3. DOVO
4. AdaBoost.M1 
5. SAMME 
6. imECOC

## Environment
List only the lowest. <br>
* OS      : Linux 
* Python  : 3 
* CUDA    : 9 
* GCC/G++ : 6.5/6.5 

## Installation
before this, please install thundergbm and thundersvm. <br>
for more information, see [ThunderGBM](https://github.com/Xtra-Computing/thundergbm), 
                      and [ThunderSVM](https://github.com/Xtra-Computing/thundersvm). <br>
```
>>> git clone https://github.com/inbliz/gpuimlearn.git
>>> cd gpuimlearn
>>> python3 setup.py install
```

## Usage

They are easy to use, and you can run them quickly. <br>
also, there is a sample in the codes folder, you can run it with the data in data folder.<br>
  
1. FocalBoost Usage <br>
FocalBoost take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
and an array y of class labels (strings or integers), of shape (n_samples):
```
# FocalBoost
>>> from gpuimlearn import FocalBoost
>>> X_train = [[1, 0.5], [2, 1], [1, 1], [2, 2], [1, 2.5], [2, 4.5]]
>>> y_train = [1, 1, 2, 2, 3, 3]
>>> clf = FocalBoost.FocalBoost()
>>> clf.fit(X_train, y_train)
FocalBoost()
```
After being fitted, the model can then be used to predict new values:
```
>>> X_test = [[1.5, 0.5], [1.5, 1.5], [1.5, 4.5]]
>>> clf.predict(X_test)
array([1, 2, 3])
```
  
2. DECOC Usage <br>
DECOC take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
and an array y of class labels (strings or integers), of shape (n_samples):
```
# DECOC
>>> from gpuimlearn import DECOC
>>> X_train = [[1, 0.5], [2, 1], [1, 1], [2, 2], [1, 2.5], [2, 4.5]]
>>> y_train = [1, 1, 2, 2, 3, 3]
>>> clf = DECOC.DECOC()
>>> clf.fit(X_train, y_train)
DECOC()
```
After being fitted, the model can then be used to predict new values:
```
>>> X_test = [[1.5, 0.5], [1.5, 1.5], [1.5, 4.5]]
>>> clf.predict(X_test)
array([1, 2, 3])
```
  
3. DOVO Usage <br>
DOVO take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
and an array y of class labels (strings or integers), of shape (n_samples):
```
# DOVO
>>> from gpuimlearn import DOVO
>>> X_train = [[1, 0.5], [2, 1], [1, 1], [2, 2], [1, 2.5], [2, 4.5]]
>>> y_train = [1, 1, 2, 2, 3, 3]
>>> clf = DOVO.DOVO()
>>> clf.fit(X_train, y_train)
DOVO()
```
After being fitted, the model can then be used to predict new values:
```
>>> X_test = [[1.5, 0.5], [1.5, 1.5], [1.5, 4.5]]
>>> clf.predict(X_test)
array([1, 2, 3])
```
  
4. AdaBoost.M1 Usage <br>
AdaBoostM1 take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
and an array y of class labels (strings or integers), of shape (n_samples):
```
# AdaBoostM1
>>> from gpuimlearn import AdaBoostM1
>>> X_train = [[1, 0.5], [2, 1], [1, 1], [2, 2], [1, 2.5], [2, 4.5]]
>>> y_train = [1, 1, 2, 2, 3, 3]
>>> clf = AdaBoostM1.AdaBoostM1()
>>> clf.fit(X_train, y_train)
AdaBoostM1()
```
After being fitted, the model can then be used to predict new values:
```
>>> X_test = [[1.5, 0.5], [1.5, 1.5], [1.5, 4.5]]
>>> clf.predict(X_test)
array([1, 2, 3])
```
  
5. SAMME Usage <br>
SAMME take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
and an array y of class labels (strings or integers), of shape (n_samples):
```
# SAMME
>>> from gpuimlearn import SAMME
>>> X_train = [[1, 0.5], [2, 1], [1, 1], [2, 2], [1, 2.5], [2, 4.5]]
>>> y_train = [1, 1, 2, 2, 3, 3]
>>> clf = SAMME.SAMME()
>>> clf.fit(X_train, y_train)
SAMME()
```
After being fitted, the model can then be used to predict new values:
```
>>> X_test = [[1.5, 0.5], [1.5, 1.5], [1.5, 4.5]]
>>> clf.predict(X_test)
array([1, 2, 3])
```
  
6. imECOC Usage <br>
imECOC take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
and an array y of class labels (strings or integers), of shape (n_samples):
```
# imECOC
>>> from gpuimlearn import imECOC
>>> X_train = [[1, 0.5], [2, 1], [1, 1], [2, 2], [1, 2.5], [2, 4.5]]
>>> y_train = [1, 1, 2, 2, 3, 3]
>>> clf = imECOC.imECOC()
>>> clf.fit(X_train, y_train)
imECOC()
```
After being fitted, the model can then be used to predict new values:
```
>>> X_test = [[1.5, 0.5], [1.5, 1.5], [1.5, 4.5]]
>>> clf.predict(X_test)
array([1, 2, 3])
```

