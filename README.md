# GPU-imLearn
a software for imbalance data classification based on GPU\n

# Install
git clone https://github.com/inbliz/gpuimlearn.git\n
cd gpuimlearn\n
python3 setup.py install\n

# Use
****DECOC****\n
from gpuimlearn import DECOC\n
decoc = DECOC.DECOC()\n
decoc.fit(x_train, y_train)\n
pre = decoc.predict(x_test)\n

