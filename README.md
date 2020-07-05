# GPU-imLearn
a software for imbalance data classification based on GPU <br>

## Environment
* List only the lowest
* OS      : Linux
* Python  : 3 
* CUDA    : 9 
* GCC/G++ : 6.5/6.5 

## Install
```
# before this, please install thundergbm and thundersvm
# for more information, see https://github.com/Xtra-Computing/thundergbm \
#                       and https://github.com/Xtra-Computing/thundersvm
git clone https://github.com/inbliz/gpuimlearn.git
cd gpuimlearn
python3 setup.py install
```

## Use
```
# DECOC
from gpuimlearn import DECOC
model = DECOC.DECOC()
model.fit(x_train, y_train)
pre = model.predict(x_test)
```

