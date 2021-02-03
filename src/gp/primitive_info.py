"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: Here are the "known" primitves. Add to all dictionaries
         to inform the code about a new primitives.

NOTES:

TODO:
"""

# define map from primitives to functions
primitive2function = {'*': 'np.multiply',
                      '+': 'np.add',
                      '-': 'np.subtract',
                      '%': 'protected_div',
                      'sin': 'np.sin',
                      'cos': 'np.cos',
                      'exp': 'protected_exp',
                      'log': 'protected_log',
                      'pow2': 'pow2',
                      'pow3': 'pow3',
                      'pow4': 'pow4',
                      'pow5': 'pow5',
                      'pow6': 'pow6'}

num_children = {'*': 2,
                '+': 2,
                '-': 2,
                '%': 2,
                'sin': 1,
                'cos': 1,
                'exp': 1,
                'log': 1,
                'pow2': 1,
                'pow3': 1,
                'pow4': 1,
                'pow5': 1,
                'pow6': 1}

# I would ideally like to get pow2(x) -> x^2.
# I am not currently setup for this though. So,
# for now we will use pow2(x) instead.
primitive2latex = {'*': '\\times',
                   '+': '+',
                   '-': '-',
                   '%': '\\%',
                   'sin': '\\sin',
                   'cos': '\\cos',
                   'exp': '\\exp',
                   'log': '\\log',
                   'pow2': '\\mathrm{pow2}',
                   'pow3': '\\mathrm{pow3}',
                   'pow4': '\\mathrm{pow4}',
                   'pow5': '\\mathrm{pow5}',
                   'pow6': '\\mathrm{pow6}'}

assert primitive2function.keys() == num_children.keys() == primitive2latex.keys(), 'Different keys between primitive2function, num_children and primitive2latex'
