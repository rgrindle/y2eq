from srvgd.utils.eval import fit_eq, get_f
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import torch
import numpy as np
import matplotlib.pyplot as plt
from eqlearner.dataset.processing.tokenization import get_string
from scipy.optimize import minimize

import cma
def RMSE(y, y_hat):
    return np.sqrt(np.mean(np.power(y-y_hat, 2)))
support = np.linspace(0.1, 3.1, 100)
num_coeffs = 2
# eq_c = 'c[0]*sin(c[1]*exp(3*x)+c[2]*exp(2*x)+c[3]*exp(x))'
eq_c = 'c[0]*sin(c[1]*x)'
f_hat = get_f(eq_c)
# loss = lambda c, x: RMSE(f_hat(c, x.T), y)
def normalize(y):
    min_ = np.min(y)
    return (y-min_)/(np.max(y)-min_)
y = normalize(f_hat(x=support, c=np.ones(num_coeffs)))
def loss(c, x):
    y_hat = f_hat(c=c, x=x)
    return RMSE(normalize(y_hat), y)
x0 = 3*np.ones(num_coeffs)
# x0 = [utils.random_from_intervals([(-10, 1), (1, 10)]) for _ in range(num_coeffs)]
print('x0', x0)
es = cma.CMAEvolutionStrategy(x0, 0.5)
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [loss(c=s, x=support) for s in solutions])
    es.disp()
es.result_pretty()
print(es.best.x)
print(es.best.f)
# soln = minimize(loss, x0, args=(support[:, None],), method='BFGS')
f = lambda x: normalize(f_hat(x=x, c=es.best.x))
support_big = np.linspace(0.1, 3.1, 100)
plt.plot(support_big, f(support_big), label='pred')
plt.plot(support, y, '.-', label='true')
plt.legend()
plt.show()
exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = get_model(device, 'cnn_epochs100_0.pt')

test_data = torch.load('test_data_int_comp0.pt', map_location=device)
x = np.arange(0.1, 3.1, 0.001)
for t in test_data:
    y = np.array(t[0])
    for i, yi in enumerate(y):
        print('(', x[i], ',', y[i], ')')
    eq = np.array(t[1])
    eq_str = get_string(eq)[5:-3]
    f = get_f(eq_str)
    coeff, rmse, f2 = fit_eq([eq_str], x, [y])
    print(rmse)
    x_big = np.arange(0.1, 3.1, 0.001)
    # plt.plot(x_big, f(x_big), '.-', label='with 1 for all coefficients')
    plt.plot(x_big, f2[0](x_big), '-', label='recomputed coefficients')
    plt.plot(x, y, '.-', label='with coefficients')
    plt.title(eq_str)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
    exit()
