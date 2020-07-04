# GPU-imLearn
a software for imbalance data classification based on GPU <br>

## Install
```
git clone https://github.com/inbliz/gpuimlearn.git
cd gpuimlearn
python3 setup.py install
```

## Use
```
# DECOC
from gpuimlearn import DECOC
decoc = DECOC.DECOC()
decoc.fit(x_train, y_train)
pre = decoc.predict(x_test)
```

