"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jul 13, 2021

PURPOSE: Can I find examples of equations that already
         exist in my current dataset that are similar?

NOTES: Determine if coefficients are similar in consistent
       of functional forms.

TODO:
"""
from equation.EquationInfix import EquationInfix
from srvgd.utils.normalize import normalize
from srvgd.utils.rmse import RMSE
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string

import torch
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_terms(sympy_expr):
    if '+' not in str(sympy_expr):
        return (sympy_expr,)
    else:
        return sympy_expr.args


def get_coeffs(ff_instance):
    coeffs = []
    for term in shared_terms:
        term = str(term)
        for term_c in get_terms(ff_instance):
            term_c = str(term_c)
            if term == '1' and 'x' not in term_c:
                coeffs.append(term_c)
                break
            elif term in term_c:
                coeffs.append(term_c.replace('*'+term, ''))
                break

    return coeffs


def get_normalized_rmse(y_true, y_pred):
    y_true_norm, true_min_, true_scale = normalize(y_true, return_params=True)
    y_pred_norm = normalize(y_pred, true_min_, true_scale)
    return RMSE(y_true_norm, y_pred_norm)


dataset = torch.load('../../datasets/dataset_train_ff1000.pt')
Y = np.squeeze([d[0].tolist() for d in dataset])
ff_list = [get_eq_string(d[1].tolist())[5:-3] for d in dataset]
print(Y.shape)

ff_instance_list = pd.read_csv('../../datasets/equations_with_coeff_train_ff1000.csv', header=None).values.flatten()
print(ff_instance_list.shape)

min_list = pd.read_csv('min_list.csv', header=None).values.flatten()

ind_list = pd.read_csv('ind_list.csv', header=None).values.flatten()
assert len(ind_list) == len(min_list)

print('max error', max(min_list))

x = sympy.symbols('x')
x_numeric = np.arange(0.1, 3.1, 0.1)

rmse_list = []
shared_rmse_list = []
for count, (ind, error) in enumerate(zip(ind_list, min_list)):
    i, j = eval(ind)
    # print(i, RMSE(Y[(i+1) % 50000], Y[j]) - error <= 10**(-20))
    index_i = (i+1) % 50000
    Yi = Y[index_i]
    Yj = Y[j]
    assert RMSE(Yi, Yj) - error <= 10**(-20)

    print(count)
    print(ff_list[index_i])
    print(ff_list[j])
    print(sympy.sympify(ff_instance_list[index_i]))
    print(sympy. sympify(ff_instance_list[j]))
    print()

    if ff_list[index_i] != ff_list[j]:
        rmse_list.append(error)

        ffi = sympy.sympify(ff_list[index_i])
        ffj = sympy.sympify(ff_list[j])
        ff_instance_i = sympy.sympify(ff_instance_list[index_i])
        ff_instance_j = sympy. sympify(ff_instance_list[j])
        shared_terms = [term for term in get_terms(ffi) if term in get_terms(ffj)]
        coeffs_i = get_coeffs(ff_instance_i)
        coeffs_j = get_coeffs(ff_instance_j)

        assert len(shared_terms) == len(coeffs_i)
        assert len(shared_terms) == len(coeffs_j)

        shared_eq_str_i = '+'.join([c+'*'+str(t) for c, t in zip(coeffs_i, shared_terms)])
        shared_eq_i = EquationInfix(shared_eq_str_i, apply_coeffs=False)
        shared_y_i = shared_eq_i.f(x_numeric)
        if type(shared_y_i) == float:
            shared_y_i = np.array([shared_y_i]*30)

        shared_eq_str_j = '+'.join([c+'*'+str(t) for c, t in zip(coeffs_j, shared_terms)])
        shared_eq_j = EquationInfix(shared_eq_str_j, apply_coeffs=False)
        shared_y_j = shared_eq_j.f(x_numeric)
        if type(shared_y_j) == float:
            shared_y_j = np.array([shared_y_j]*30)

        eq_true_i = EquationInfix(ff_instance_list[index_i], apply_coeffs=False)
        rmse_i = get_normalized_rmse(y_true=eq_true_i.f(x_numeric),
                                     y_pred=shared_y_i)
        eq_true_j = EquationInfix(ff_instance_list[j], apply_coeffs=False)
        rmse_j = get_normalized_rmse(y_true=eq_true_j.f(x_numeric),
                                     y_pred=shared_y_j)

        # plt.plot(x_numeric, shared_y_i, label='predi')
        # plt.plot(x_numeric, eq_true_i.f(x_numeric), label='truei')
        # plt.plot(x_numeric, shared_y_j, label='predj')
        # plt.plot(x_numeric, eq_true_j.f(x_numeric), label='truej')
        # plt.legend()
        # plt.show()

        shared_rmse_list.extend([rmse_i, rmse_j])

print(len(rmse_list), len(shared_rmse_list))
plt.boxplot([rmse_list],    # shared_rmse_list],
            labels=['75 comparisons between numerically similar\nfunctional form instances'])    # , 'Shared terms of ff instances'])
plt.yscale('log')
plt.ylabel('Numeric cost')
plt.savefig('shared_term_rmse_pres.pdf')
