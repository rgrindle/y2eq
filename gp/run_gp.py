"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 26, 2021

PURPOSE: Run genetic programming.

NOTES:

TODO:
"""
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x_train = np.arange(0.1, 3.1, 0.1)[:, None]
x_test = np.random.uniform(0.1, 3.0, 30)[:, None]
x_test_ext = np.arange(3.1, 6.1, 0.1)[:, None]
f = lambda x: 2.75*x[0]**2
y_train = f(x_train.T)
y_test = f(x_test.T)
y_test_ext = f(x_test_ext.T)


def safe_exp(x):
    """safe_exp(x) = exp(x) if x < 100 else 0"""
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.)


exp = make_function(function=safe_exp,
                    name='exp',
                    arity=1)

gp = SymbolicRegressor(verbose=1,
                       function_set=['add', 'sub', 'mul', exp, 'sin', 'log'],
                       const_range=(-3.0, 3.0),
                       metric='rmse',
                       population_size=100,
                       generations=100,
                       p_crossover=0.9,
                       p_subtree_mutation=0.01,
                       p_hoist_mutation=0.01,
                       p_point_mutation=0.01,
                       p_point_replace=0.05,
                       init_depth=(2, 6),
                       init_method='half and half',
                       tournament_size=20)
gp.fit(x_train, y_train)
y_pred = gp.predict(x_test)
print(gp._program)

plt.figure()
plt.plot(x_train, y_train, 'o', label='true (train)')
plt.plot(x_test, y_test, 'o', label='true (test)')
plt.plot(x_test, y_pred, 'o', ms=3, label='pred (test)')
plt.plot(x_test_ext, y_test_ext, 'o', label='true (test ext)')
plt.plot(x_test_ext, gp.predict(x_test_ext), 'o', ms=3, label='pred (test ext)')
plt.legend()
plt.show()
