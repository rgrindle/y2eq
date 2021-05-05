"""
AUTHOR: Ryan Grindle

LAST MODIFIED: April 1, 2021

PURPOSE: Update functions used in eqlearner.dataset.processing.tokenization
         to get a version of pipeline (which I am calling tokenize_eq) that
         works on strings rather than dictionaries.

NOTES: Also have changed default_dict to token_map and created
       inverse_token_map. Also, have renamed get_string to
       get_eq_string

TODO:
"""
import numpy as np

import tokenize
from io import BytesIO


token_map = {'': 0, 'x': 1, 'sin': 2, 'exp': 3, 'log': 4, '(': 5, ')': 6, '**': 7, '*': 8, '+': 9,
             '/': 10, 'E': 11, 'START': 12, 'END': 13, 'sqrt': 14, '-': 15}
max_val = max(list(token_map.values()))
numbers = {str(n): max_val+n for n in range(1, 10)}
token_map = {**token_map, **numbers}
inverse_token_map = {token_map[key]: key for key in token_map}


token_map_2d = {'': 0, 'x': 1, 'sin': 2, 'exp': 3, 'log': 4, '(': 5, ')': 6, '**': 7, '*': 8, '+': 9,
                '/': 10, 'E': 11, 'START': 12, 'END': 13, 'sqrt': 14, '-': 15,
                'x0': 16, 'x1': 17}
max_val_2d = max(list(token_map_2d.values()))
numbers_2d = {str(n): max_val_2d+n for n in range(1, 10)}
token_map_2d = {**token_map_2d, **numbers_2d}
inverse_token_map_2d = {token_map_2d[key]: key for key in token_map_2d}


# There are two versiont that go with the following map. In both,
# NN is expected to output actual numbers for the coefficient. In version 1,
# The number is output as sequences of tokens (e.g. [1, ., 0, 2] for 1.02).
# In version 2, the NN outputs c and then another unit represented the
# coefficient value outptus an actual number. In both versions only one additional
# output unit is necessary for the classification portion so both c and . are
# represented by 16.
token_map_with_coeffs = {'': 0, 'x': 1, 'sin': 2, 'exp': 3, 'log': 4, '(': 5, ')': 6, '**': 7, '*': 8, '+': 9,
                         '/': 10, 'E': 11, 'START': 12, 'END': 13, 'sqrt': 14, '-': 15, 'c': 16, '.': 16}
max_val_with_coeffs = max(list(token_map_with_coeffs.values()))
numbers_with_coeffs = {str(n): max_val_with_coeffs+n+1 for n in range(10)}
token_map_with_coeffs = {**token_map_with_coeffs, **numbers_with_coeffs}
inverse_token_map_with_coeffs = {token_map_with_coeffs[key]: key for key in token_map_with_coeffs}


def numberize_tokens(tokens, two_d, include_coeffs):
    assert not (two_d and include_coeffs)
    if two_d:
        mapping = token_map_2d
    elif include_coeffs:
        mapping = token_map_with_coeffs
    else:
        mapping = token_map

    return [mapping[di] for di in tokens]


def extract_tokens(string, group_minus_signs=False):
    string = string.replace('START', '').replace('END', '')
    extracted_tokens = ['START']+[elem.string for elem in tokenize.tokenize(BytesIO(string.encode('utf-8')).readline)]+['END']
    extracted_tokens.remove('utf-8')
    extracted_tokens = [e for e in extracted_tokens if e != '']

    if group_minus_signs:
        new_extracted_tokens = []
        skip_next = False
        for i, token in enumerate(extracted_tokens):
            if skip_next:
                skip_next = False
                continue

            if token == '-':
                # Minus is not a primitive, so
                # group them with coeff always
                skip_next = True
                new_extracted_tokens.append('-'+extracted_tokens[i+1])
            else:
                new_extracted_tokens.append(token)

        extracted_tokens = new_extracted_tokens

    return extracted_tokens


def tokenize_eq(eq_str, two_d=False):
    extracted_tokens = extract_tokens(eq_str)
    return numberize_tokens(extracted_tokens, two_d)


def get_eq_string(numberized_tokens, two_d=False, include_coeffs=False,
                  coeff_output=None, include_coeff_values=False):
    assert not (two_d and include_coeffs)
    if two_d:
        inv_mapping = inverse_token_map_2d
    elif include_coeffs:
        inv_mapping = inverse_token_map_with_coeffs
    else:
        inv_mapping = inverse_token_map

    eq_list = [inv_mapping[digit] for digit in numberized_tokens]

    if include_coeff_values:
        # If here, then the coefficients are held in
        # coeff_output, so put them in the equation string.
        assert include_coeffs
        assert coeff_output is not None

        coeff_output.insert(0, float('NaN'))
        assert len(eq_list) == len(coeff_output), 'len(eq_list) = '+str(len(eq_list))+', len(coeff_output) = '+str(len(coeff_output))

        for i, (token, coeff) in enumerate(zip(eq_list, coeff_output)):
            if token in ('c', '.'):
                eq_list[i] = str(np.round(coeff, 3))

    return ''.join(eq_list)


if __name__ == '__main__':
    import numpy as np

    tokens = ['0', '.', '1', '2', '3', '*', 'x']
    n = numberize_tokens(tokens, two_d=False, include_coeffs=True)

    print('tokens', tokens)
    print('numberized_tokens', n)

    def test_extract_tokens():
        eq = '0.812*sin(-0.073*x)**6+0.894*sin(-1.568*x)**2+-1.734*sin(-0.364*x)+-0.779*exp(-0.775*sin(-2.806*x)**3+-0.098*sin(-0.143*x)+2.740*1)'
        e_tokens = extract_tokens(eq, group_minus_signs=True)
        print(eq)
        print(e_tokens)
        assert np.all('-' != np.array(e_tokens))

    test_extract_tokens()
