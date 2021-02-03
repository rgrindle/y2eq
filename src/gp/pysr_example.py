import numpy as np
from pysr import pysr, best

# Dataset
X = 2*np.random.randn(100, 5)
# y = 2*np.cos(X[:, 3]) + X[:, 0]**2 - 2
y = 0.706*np.sin(-0.205*X[:, 0])**6+-0.315*np.sin(0.021*X[:, 0])**2+0.878*np.sin(-1.690*X[:, 0])+2.446*np.exp(-0.543*np.sin(1.048*X[:, 0])**3+-2.459*np.sin(0.127*X[:, 0])+-2.070*1)

# Learn equations
equations = pysr(X, y, niterations=5,
                 binary_operators=["plus", "mult", 'pow'],
                 unary_operators=["exp", "sin", 'log'])  # Pre-defined library of operators (see https://pysr.readthedocs.io/en/latest/docs/operators/)

# (you can use ctl-c to exit early)

print(best(equations))
