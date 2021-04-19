"""
AUTHOR: Ryan Grindle

LAST MODIFIED: April 6, 2021

PURPOSE: BFGS has difficulty determining coefficients for
         some of the fuctional forms I previously picked at
         random. Since I am not trying to test BFGS's ability
         I will pick the 1000 functional forms with lowest
         RMSE after using BFGS.

NOTES:

TODO:
"""
from srvgd.utils.normalize import normalize
from srvgd.utils.eval import RMSE
from equation.EquationInfix import EquationInfix

import numpy as np
import pandas as pd

import json


class FFObject:
    def __init__(self, ff):
        self.x = np.arange(0.1, 3.1, 0.1)
        self.ff = ff
        self.EquationInfix = EquationInfix(ff)

        self.generate_equations()

    def generate_equations(self):
        num_eq_generated = 0
        num_attempts = 0
        self.eq_list = []
        self.y_list = []
        self.rmse_list = []
        while num_eq_generated < 51:
            coeffs = np.random.uniform(-3, 3, self.EquationInfix.num_coeffs)
            coeffs = np.round(coeffs, 3)
            y = self.EquationInfix.f(coeffs, self.x)

            if (not np.any(np.isnan(y))) and (np.all(np.abs(y) <= 1000)):
                num_eq_generated += 1
                self.y_list.append(y)
                self.eq_list.append(self.EquationInfix.place_exact_coeffs(coeffs))
                self.EquationInfix.fit(y)
                normalized_rmse = self.get_normalized_rmse(y)
                self.rmse_list.append(normalized_rmse)
            else:
                num_attempts += 1

            print(',', end='', sep='', flush=True)

            if num_attempts > 100:
                print('RAN OUT OF ATTEMPTS')
                self.y_list = np.array(51*[30*[np.nan]])
                self.eq_list = 51*[None]
                self.rmse_list = [np.inf]
                break
        print()
        print('Random coefficient were unacceptible', num_attempts, 'times')

    def get_normalized_rmse(self, y_true):
        y_true_norm, true_min_, true_scale = normalize(y_true, return_params=True)
        y_pred = self.EquationInfix.f(c=self.EquationInfix.coeffs, x=self.x)
        y_pred_norm = normalize(y_pred, true_min_, true_scale)
        return RMSE(y_true_norm, y_pred_norm)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int,
                        help='The script will work with ff starting '
                             'at index of start*num_ff_to_do and end at '
                             'start*(num_ff_to_do+1).',
                        required=True)
    parser.add_argument('--num_ff_to_do', type=int,
                        default=100,
                        help='The number of ff to do per run of this '
                             'script')
    args = parser.parse_args()

    np.random.seed(1234)

    unique_ff_list = pd.read_csv('unique_ff_list.csv', header=None).values.flatten()

    FF_list = []
    rmse_avg_list = []
    for i, ff in enumerate(unique_ff_list[args.start:args.start+args.num_ff_to_do]):
        FF = FFObject(ff)
        FF_list.append(FF)
        rmse_avg_list.append(np.mean(FF.rmse_list))
        print(i, rmse_avg_list[-1], ff)
        print()

    # sorted_indices = np.argsort(rmse_avg_list)
    # sorted_rmse_avg_list = np.array(rmse_avg_list)[sorted_indices]
    # sorted_ff_list = np.array(unique_ff_list)[sorted_indices]
    ff_list = [FF.ff for FF in FF_list]

    print('Saving functional forms...', end='', flush=True)
    pd.DataFrame([ff_list, rmse_avg_list]).T.to_csv('ff_with_rmse_avg_start{}.csv'.format(args.start),
                                                    index=False, header=['ff', 'rmse_avg'])
    print('done')

    y_list = []
    eq_list = []
    for FF in np.array(FF_list):
        y_list.extend([y.tolist() for y in FF.y_list])
        eq_list.extend(FF.eq_list)

    print('y', len(y_list), len(y_list[0]))

    if eq_list[0] is not None:
        print('eq', len(eq_list), len(eq_list[0]))

    with open('y_list_start{}.json'.format(args.start), 'w') as file:
        json.dump(y_list, file)

    with open('eq_list_start{}.json'.format(args.start), 'w') as file:
        json.dump(eq_list, file)
