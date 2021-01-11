"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 7, 2021

PURPOSE: Generate train dataset of 50000 observations where
         each observation is a symbolic regression problem/answer

NOTES: Modified from SeqSeqModel.ipynb

TODO:
"""
from DatasetCreatorRG import DatasetCreatorRG
from eqlearner.dataset.processing import tokenization

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import random
from sympy import sin, log, exp, Symbol

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    from tensor_dataset import TensorDatasetGPU as TensorDataset  # noqa: F401
else:
    from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

dataset_size = 5

scaler = MinMaxScaler()
x = Symbol('x', real=True)
basis_functions = [x, sin, log, exp]
support = np.arange(0.1, 3.1, 0.1)
DC = DatasetCreatorRG(basis_functions,
                      max_linear_terms=1,
                      max_binomial_terms=1,
                      max_compositions=1,
                      max_N_terms=0,
                      division_on=False,
                      random_terms=True,
                      constants_enabled=True,
                      constant_intervals_ext=[(-3, 1), (1, 3)],
                      constant_intervals_int=[(1, 3)])

# for _ in range(3):
fun, dictionary, dictionary_cleaned = DC.generate_fun()
print(fun)

random.seed(SEED)
# np.random.seed(SEED)

fun, dictionary, dictionary_cleaned = DC.generate_fun()
print(fun)

random.seed(SEED)
# np.random.seed(SEED)

fun, dictionary, dictionary_cleaned = DC.generate_fun()
print(fun)
exit()

dataset_input = []
dataset_output = []
fun_list = []
count = 0
cond = True
while count < dataset_size:
    fun, dictionary, dictionary_cleaned = DC.generate_fun()
    print(fun)
    if fun != 0:
        Y = DC.evaluate_function(support, fun, X_noise=False)
        # numeric = np.array([support, Y])
        if np.abs(np.max(Y)) < 1000:
            dataset_input.append(Y)
            tokenized_eq = tokenization.pipeline([dictionary_cleaned])[0]
            dataset_output.append(torch.Tensor(tokenized_eq))
            fun_list.append(fun)
            count += 1
        else:
            print('function out of bounds')
    else:
        print('all zeros. fun =', fun)
print(np.array(dataset_input).shape)
print()
for fun in fun_list:
    print(fun)
# scaler.fit(np.array(dataset_input).T)
# x_train_n = scaler.transform(np.array(dataset_input).T)
# x_train_n = torch.Tensor(x_train_n)
# l = [len(y) for y in dataset_output]
# q = np.max(l)
# dataset_output_p = torch.zeros(len(dataset_output), q)
# for i, y in enumerate(dataset_output):
#     dataset_output_p[i, :] = torch.cat([y, torch.zeros(q-y.shape[0])])
# dataset = TensorDataset(x_train_n.T, dataset_output_p.long())

# x_test = []
# y_test = []
# cnt = 0
# cond = True
# while cond:
#     string, dictionary = fun_generator.generate_batch(support,1, X_noise=False, Y_noise=0)
#     if np.all(string[0][1] == 0) == False:
#         if np.max(string[0][1]) < 1000 and np.min(string[0][1]) > -1000 and tokenization.get_string(tokenization.pipeline(dictionary)[0])[-1] != '+': 
#             x_test.append(string[0][1])
#             y_test.append(torch.Tensor((tokenization.pipeline(dictionary)[0])))
#             cnt += 1
#     if cnt == test_eqs:
#         cond = False
# scaler = MinMaxScaler()
# x_test_n = scaler.fit_transform(np.array(x_test).T)
# x_test_n = torch.Tensor(x_test_n)
# l = [len(y) for y in y_test]
# q = np.max(l)
# y_test_p = torch.zeros(len(y_test), q)
# for i, y in enumerate(y_test):
#     y_test_p[i, :] = torch.cat([y, torch.zeros(q-y.shape[0])])
# test_data = TensorDataset(x_test_n.T, y_test_p.long())
