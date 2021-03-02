from srvgd.eval_scripts.get_x_y import get_x_y

import torch
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eq_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).values.flatten()

# x_int = np.arange(0.1, 3.1, 0.1)
x_ext = np.arange(3.1, 6.1, 0.1)

get_x_y(x_int_type='uniform',
        x_int_num=30,
        x_ext=x_ext,
        eq_list=eq_list)
