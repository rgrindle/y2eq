from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from srvgd.utils.eval import apply_coeffs

import torch
import pandas as pd
import matplotlib.pyplot as plt

rmse = pd.read_csv('01_rmse_all.csv')['rmse_int']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eq_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).values.flatten()

dataset = torch.load('../datasets/dataset_test_ff1000.pt', map_location=device)
ff_list = [get_eq_string(d[1].tolist())[5:-3] for d in dataset]
num_coeffs_list = [apply_coeffs(ff)[1] for ff in ff_list]

plt.plot(num_coeffs_list, rmse, '.')
plt.xlabel('Numer of Coefficients in ff')
plt.ylabel('RMSE with fitted Coefficients')
plt.yscale('log')
plt.show()
