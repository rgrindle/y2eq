"""
AUTHOR: Ryan Grindle (modified)

LAST MODIFIED: Jan 11, 2021

PURPOSE: Remove use of set class, so that dataset
         generation is consist if consistent seed is used.

NOTES: Original code here: https://github.com/SymposiumOrganization/EQLearner/blob/main/lib/src/eqlearner/dataset/univariate/eqgenerator.py
       commit dfe0c7dd54584b4987fcbc51d05e117133a68d95

TODO:
"""
from srvgd.updated_eqlearner.EquationStructuresRG import EquationStructuresRG as EquationStructures

import bisect
import numpy as np
# from sympy.utilities.lambdify import lambdify
import sympy

# import random
from typing import List
import itertools
import os
import pdb
import copy
from functools import reduce     # Valid in Python 2.6+, required in Python 3
import operator

# -----------------------------------------
# Modified functions - BELOW
# -----------------------------------------


def Binomial_single(total_combinations, expression: List, priority_list, symbol, constant_interval=[(1, 1)]):
    """set used for set minus, replaced with list comprehension"""
    curr = priority_list
    candidate_keys = total_combinations.keys()
    try:
        chosen = np.random.choice([key for key in candidate_keys if key not in curr])
    except:
        raise ValueError('The number of linear functions is bigger than the number of linear terms')
    i = bisect.bisect(priority_list, chosen)
    priority_list.insert(i, chosen)
    elem_with_constant = constant_adder_binomial(total_combinations[chosen], symbol, constant_interval)
    expression.insert(i, elem_with_constant)
    return expression, priority_list


def N_single(basis_functions, fun_list, priority_list, n, begin=3):
    """set used for set minus, replaced with list comprehension"""
    curr = priority_list     # watch out exp(x)^2 == exp(x^2)
    ordered = order_assigner(N_creator(basis_functions, n, begin))
    candidate_keys = ordered.keys()
    some_keys = [key for key in candidate_keys if key not in curr]
    if some_keys:
        chosen = np.random.choice(some_keys)
        i = bisect.bisect(priority_list, chosen)
        priority_list.insert(i, chosen)
        fun_list.insert(i, ordered[chosen])
    return fun_list, priority_list


# def Division_single(basis_functions,n):
#     """Changed from random to np.random"""
#     #We said that we are just creating a single instane of division. Hence priority not really necessary. 
#     candidate = np.random.choice(basis_functions)
#     numerator = []
#     denominator = []
#     while numerator == denominator:
#         numerator = []
#         denominator = []
#         poly = []
#         priority = []
#         for i in range(n):
#             numerator, priority =  N_single([candidate],numerator, priority,6,begin=0)
#             denominator, priority = N_single([candidate],denominator, priority, 6,begin=0)
#     symbolic_numerator = expression_creator({"numerator": numerator})
#     symbolic_denominator = expression_creator({"denominator": denominator})
#     division = symbolic_numerator/symbolic_denominator 
#     #logger.info("Create raw Division Term {}".format(str(division)))
#     division, priority = eliminate_infity_term(division, 999)
#     return [division], priority


# def eliminate_random_terms(expression,probability):
#     """Changed from random to np.random"""
#     for key in final_expression:
#         to_drop = np.random.random_integers(0,10) > probability*10
#         key.remove()
#     return final_expression

# -----------------------------------------
# Modified functions - ABOVE
# -----------------------------------------

# -----------------------------------------
# Original (unmodefied) functions - BELOW
# -----------------------------------------


def Const_term():
    return 1, 1


def basis_and_symbol_joiner(basis_function,symbol, constant_interval=[(1,1)]): 
    if type(basis_function) == sympy.core.symbol.Symbol:
            symbol = basis_function
            return basis_function
    else:
        if type(basis_function) != sympy.core.function.FunctionClass:
            raise(TypeError, "Basis functions must be func or symbol")
        else: 
            try:
                c = random_from_intervals(constant_interval)
                res = basis_function(symbol*c)
                return res
            except:
                print("Something wrong happended")
                pdb.set_trace()


def constant_adder_binomial(elements: List,symbol, constant_interval=[(1,1)]):
    assert len(elements) == 2
    with_constants = []
    for basis_function in elements:
        with_constants.append(basis_and_symbol_joiner(basis_function,symbol,constant_interval))
    return with_constants[0]*with_constants[1]


def order_assigner(list_of_terms):
        return { k:i for k, i in  enumerate(list_of_terms)}


def polynomial_single(tracker,expression: List, raw: List,symbol, constant_interval=[(1,1)]):
    # Add a linear function to the current set. It returns two:
    # One-List: the ordered set of basis functions
    # Second-List: number that represents the priority of the basis function
    # curr = set(priority_list)
    # candidate_keys = set(total_combinations.keys())
    # try:
    #     chosen = random.choice(list(candidate_keys-curr))
    # except:
    #     raise ValueError('The number of linear functions is bigger than the number of linear terms')
    # i = bisect.bisect(priority_list, chosen)
    # priority_list.append(chosen)
    chosen = tracker.get_equation(drop=0)
    raw.append(chosen)
    elem_with_constant = EquationStructures.polynomial_joiner(chosen, symbol, constant_interval, constant_interval)
    expression.append(elem_with_constant)
    return expression, raw


def N_creator(basis_functions:list(),n,begin):
    pr_basis = []
    factors = []
    for n in range(begin,n+1):
        if n == 0:
            pr_basis.append(1)  
            continue
        for i in itertools.combinations_with_replacement(basis_functions, n):
            pr_basis.append(reduce(operator.mul, i))
    return pr_basis


# def Composite_creator(path_to_composite,expr):
#     check_if_exist()
#     return Composite_creator

# def Join_expression(*args):
#     total_coeff = sum([count_depth_dictionary([entry]) for entry in args])
#     coeffs = np.random.random_sample(len(total_coeff))
#     res = join_expression(args)
#     expression = 0
#     for idx ,curr_item in enumerate(myprint):
#         expression = coeffs[idx]*curr_item
#     return expression


def Composition_single(tracker,expression: List,raw: List,symbol,constant_interval=[1,1]):
    #ordered = order_assigner(raw_basis_functions).keys()
    #parent_keys = [key*1000 for key in ordered]
    #parent_key = random.choice(list(ordered)) #Can be also equal, not really imporant
    #child_basis = random.sample(list(set(basis_functions)-{raw_basis_functions[parent_key](symbol)}),2)
    #raw_basis_functions_set = set(raw_basis_functions)
    #raw_basis_functions_set.add(symbol)
    #child_basis = random.sample(list(raw_basis_functions_set-{raw_basis_functions[parent_key]}),2)
    chosen = tracker.get_equation(drop=2)
    raw.append(chosen)
    curr = EquationStructures.composition_joiner(chosen, symbol, constant_interval, constant_interval) 
    expression.append(curr)

    # if 0:
    #     tmp = Binomial_single(EquationStructures(child_basis).binomial,[],[],symbol, constant_interval= [(1,1)])  
    #     res  = tmp[0][0] + res
    #     pr = tmp[1][0] + pr
    #final_keys = parent_key*1000 + pr  
    #composition = raw_basis_functions[parent_key](res)
    #composition, final_keys = eliminate_infity_term(composition,final_keys)
    
    #compositions.append(composition)
    return expression, raw


def expression_creator(one_dictionary):
    total_coeff = 0
    expression = 0
    for key in one_dictionary.keys():
        total_coeff = len(one_dictionary[key]) + total_coeff
        #coeffs = np.random.random_sample(total_coeff)*0 + 1 #No constant
        for idx ,curr_item in enumerate(one_dictionary[key]):
            expression = curr_item + expression
            #expression = coeffs[idx]*curr_item + expression
    return expression


# def expression_joiner(*dictionaries):
#     final_expression = {dictionaries}
#     return final_expression
# def eliminate_infity_term(expression, priority_list):
#     if expression == zoo:
#         return 0, 0
#     else:
#         return expression, priority_list


# def function_evaluator(support,expression):
#     function = lambdify(x, result)
#     res_dictionary['GT_Symbolic'] = expression
#     res_dictionary['y'] = y
#     res_dictionary['coeff'] = coeff
#     res_dictionary['basis_fun'] = basis_functions
#     res_dictionary['x'] = support['support']
#     y = np.array([function(i) for i in support['support']])
#     return res_dictionary


def Noise_adder_y(dataset: dict, var = 1.0):
    dataset = copy.deepcopy(dataset)
    for eq in dataset:
        eq['y'] = eq['y'] + np.random.normal(scale=np.sqrt(var), size=len(eq['y']))
    return dataset


# def Noise_adder_x(dataset: dict, var = 0.025):
#     dataset = copy.deepcopy(dataset)
#     for eq in dataset:
#         eq['x'] = eq['x'] + np.random.normal(scale=np.sqrt(var), size=len(eq['x']))
#         result = np.dot(eq['basis_fun'],eq['coeff'])
#         function = lambdify(x, result)
#         eq['y'] = np.array([function(i) for i in eq['x']])
#     return dataset


# def count_depth_dictionary(d):
#     return sum([count(v)+1 if isinstance(v, dict) else 1 for v in d.values()])


# def entry_returner(d):
#     for k, v in d.items():
#         if isinstance(v, dict):
#             myprint(v)
#         yield ("{0} : {1}".format(k, v))


def save_generated_data(data,dir = r"C:\Users\lbg\OneDrive - CSEM S.A\Bureau\Pytorch\NEW_EQ_LEARN\Data"):
    name_key = list(data.keys())[0]
    #Save support points 
    path = os.path.join(dir,name_key)
    np.save(path,data)


def random_from_intervals(intervals):   # intervals is a sequence of start,end tuples
    """For some reason random.uniform was not behaving consistently
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
