from srvgd.architecture.transformer.y2eq_transformer import y2eq_trans_model
from srvgd.utils.eval import get_eq_y2eq_transformer
from srvgd.utils.rmse import RMSE
from equation.EquationInfix import EquationInfix
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from srvgd.utils.normalize import normalize

import torch
import numpy as np
import pandas as pd


x = np.linspace(-1, 1, 20)
test_dataset = torch.load('../../datasets/dataset_msr_test.pt')

inputs = [d[0] for d in test_dataset]
outputs = [get_eq_string(d[1].tolist())[5:-3] for d in test_dataset]
true_eq_list = [EquationInfix(out, apply_coeffs=False) for out in outputs]
y_list = [eq.f(x) for eq in true_eq_list]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = torch.load('../../models/BEST_y2eq_comp_msr.pt', map_location=device)
y2eq_trans_model.load_state_dict(model_dict['state_dict'])

predicted_data = []

for i, inp in enumerate(inputs):

    predicted = get_eq_y2eq_transformer(sentence=inp,
                                        model=y2eq_trans_model,
                                        device=device)
    predicted_data.append(predicted[5:-3])
    print(i, predicted_data[-1])

# get y
f_list = [EquationInfix(p, apply_coeffs=False) for p in predicted_data]
y_hat_list = [eq.f(x) for eq in f_list]
y_hat_norm_list = []
y_norm_list = []
for y, y_hat in zip(y_list, y_hat_list):
    y_norm, m, s = normalize(y, return_params=True)
    y_hat_norm = normalize(y_hat, m, s)
    y_norm_list.append(y_norm)
    y_hat_norm_list.append(y_hat_norm)

rmse_list = [RMSE(y_hat, y) for y_hat, y in zip(y_hat_norm_list, y_norm_list)]
pd.DataFrame(rmse_list).to_csv('y2eq_data.csv', index=False, header=None)
