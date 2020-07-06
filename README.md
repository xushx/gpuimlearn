# GPU-imLearn
a software for imbalance data classification based on GPU. <br>

## Document Description
* codes : the code work.
* data : 2 sample datasets, which are '.mat' format.
* docs : some related papers.

## Software Contents
There are 3 main parts in this work, you can see them in the codes folder. <br>
* cpu 
6 multi-class imbalanced classification algorithms based on CPU
* gpuimlearn <recommend>
6 multi-class imbalanced classification algorithms based on GPU, base classifications are implemented by CUDA.
* pytorch 
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
  
1. FocalBoost Usage 
As other classifiers, FocalBoost take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
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
  
2. DECOC Usage 
As other classifiers, DECOC take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
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
  
3. DOVO Usage 
As other classifiers, DOVO take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
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
  
4. AdaBoost.M1 Usage 
As other classifiers, AdaBoostM1 take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
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
  
5. SAMME Usage 
As other classifiers, SAMME take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
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
  
6. imECOC Usage 
As other classifiers, imECOC take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, <br>
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

