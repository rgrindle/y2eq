"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 11, 2021

PURPOSE: Remove use of set class, so that dataset
         generation is consist if consistent seed is used.

NOTES:

TODO:
"""
import eqlearner.dataset.utils as utils
from eqlearner.dataset.univariate.equationstructure import EquationStructures

import itertools
import numpy as np
import sympy
from collections import defaultdict


class EquationStructuresRG(EquationStructures):

    @classmethod
    def _polynomial_enumerator(cls, basis_funtion, order=6, drop_list=[], prefix=()):
        """Set was used to get unique elements. Replaced with for loop with if statement"""
        combinations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for basis_fun in basis_funtion:
            prod = itertools.product([basis_fun, 0], repeat=order)
            unique_prod = []
            for p in prod:
                if p not in unique_prod:
                    unique_prod.append(p)
            for x in unique_prod:
                for c in cls._constant_enumerator():
                    if all([i == 0 for i in x]) or x in drop_list:
                        continue
                    combinations[basis_fun][utils.count_element(x)][c].append([c] + list(x))
        return combinations

    @staticmethod
    def polynomial_joiner(candidate, symbol, const_interval_ext=[(1, 1)], constant_interval_int=[(1, 1)]):
        """new random_from_intervals"""
        res = candidate[0]*random_from_intervals(const_interval_ext)
        candidate_pol = candidate[1:]
        for idx, elem in enumerate(candidate_pol, 1):
            if elem == 0:
                continue
            elif type(elem) == sympy.core.symbol.Symbol:
                external = random_from_intervals(const_interval_ext)
                res = external*elem**(idx) + res
            elif type(elem) == sympy.core.function.FunctionClass:
                interal = random_from_intervals(constant_interval_int)
                external = random_from_intervals(const_interval_ext)
                tmp = external*elem(interal*symbol)**(idx)
                res = tmp + res
        return res


def random_from_intervals(intervals):   # intervals is a sequence of start,end tuples
    """For some readon random.uniform was not behaving consistently
    with random seed, so I changed it."""
    total_size = sum(end-start for start, end in intervals)
    n = np.random.uniform(0, total_size)
    if total_size > 0:
        for start, end in intervals:
            if n < end-start:
                return round(start + n, 3)
            n -= end-start
    else:
        return 1
