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


def numberize_tokens(tokens, two_d):
    if two_d:
        return [token_map_2d[di] for di in tokens]
    else:
        return [token_map[di] for di in tokens]


def extract_tokens(string):
    string = string.replace('START', '').replace('END', '')
    extracted_tokens = ['START']+[elem.string for elem in tokenize.tokenize(BytesIO(string.encode('utf-8')).readline)]+['END']
    extracted_tokens.remove('utf-8')
    return [e for e in extracted_tokens if e != '']


def tokenize_eq(eq_str, two_d=False):
    extracted_tokens = extract_tokens(eq_str)
    return numberize_tokens(extracted_tokens, two_d)


def get_eq_string(numberized_tokens, two_d=False):
    if two_d:
        return ''.join([inverse_token_map_2d[digit] for digit in numberized_tokens])
    else:
        return ''.join([inverse_token_map[digit] for digit in numberized_tokens])
