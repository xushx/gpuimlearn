# GPU-imLearn
a software for imbalance data classification based on GPU <br>

## Install
'''
git clone https://github.com/inbliz/gpuimlearn.git <br>
cd gpuimlearn <br>
python3 setup.py install <br>
'''

## Use
'''
**DECOC** <br>
from gpuimlearn import DECOC <br>
decoc = DECOC.DECOC() <br>
decoc.fit(x_train, y_train) <br>
pre = decoc.predict(x_test) <br>
'''

