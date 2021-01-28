"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 22, 2021

PURPOSE: Update functions used in eqlearner.dataset.processing.tokenization
         to get a version of pipeline (which I am calling tokenize_eq) that
         works on strings rather than dictionaries.

NOTES:

TODO:
"""

import tokenize
from io import BytesIO


def default_map():
    default_map = {'x': 1, 'sin': 2, 'exp': 3, 'log': 4, '(': 5, ')': 6, '**': 7, '*': 8, '+': 9,
                   '/': 10, 'E': 11, 'START': 12, 'END': 13, 'sqrt': 14, '-': 15}
    max_val = max(list(default_map.values()))
    numbers = {str(n): max_val+n for n in range(1, 10)}
    default_map = {**default_map, **numbers}
    return default_map


def numberize_tokens(tokens):
    mapping = default_map()
    return [mapping[di] for di in tokens]


def extract_tokens(string):
    string = string.replace('START', '').replace('END', '')
    extracted_tokens = ['START']+[elem.string for elem in tokenize.tokenize(BytesIO(string.encode('utf-8')).readline)]+['END']
    extracted_tokens.remove('utf-8')
    return [e for e in extracted_tokens if e != '']


def tokenize_eq(eq_str):
    extracted_tokens = extract_tokens(eq_str)
    return numberize_tokens(extracted_tokens)
