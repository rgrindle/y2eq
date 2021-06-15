"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 3, 2021

PURPOSE: I want to show an example of BFGS fitting
         the "wrong" functional form to data where
         the solution is NOT to zero out certain
         coefficients.

NOTES:

TODO:
"""
from srvgd.utils.rmse import RMSE

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


x = np.arange(0.1, 3.1, 0.1)

ff_pred = lambda c, x: c[0]*np.sin(c[1]*x + c[2]*1) + c[3]*1
num_coeffs = 4

eq_true = lambda x: 0.1*np.exp(x)
y_true = eq_true(x)

init_guess = np.ones(num_coeffs)


def loss(c, x):
    y_hat = ff_pred(c, x).flatten()
    return RMSE(y_hat=y_hat, y=y_true)


res = minimize(loss, init_guess, args=(x,),
               bounds=[(-3, 3)]*num_coeffs,
               method='L-BFGS-B')
coeffs = res.x
print('coeffs', coeffs)

y_pred = ff_pred(coeffs, x)
print('RMSE', RMSE(y_hat=y_pred, y=y_true))

plt.plot(x, y_true, '.-', label='true')
plt.plot(x, y_pred, ',-', label='pred')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
