import numpy as np
import pandas as pd

import json

file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates_660'
# file_endname = '_epochs100_0'
with open('01_valid_eq{}.json'.format(file_endname), 'r') as json_file:
    valid_equations = json.load(json_file)

unique_valid_eqs = np.unique(list(valid_equations.values()))
print(len(unique_valid_eqs))
for u in unique_valid_eqs:
    print(u)

# ff_list = pd.read_csv('modifying_functional_forms_list00.output', header=None).values.flatten()
ff_list = pd.read_csv('get_function_forms_test.csv').iloc[:, 1].values.flatten()
print(ff_list)

print('List of ff not in ff_list')
count = 0
for u in unique_valid_eqs:
    if u not in ff_list:
        print(u)
        count += 1
print(count, 'ff output by NN that are not in train dataset')
print('out of', len(unique_valid_eqs))
