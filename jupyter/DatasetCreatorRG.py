"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 11, 2021

PURPOSE: This where I have made changes to
         eqlearner.dataset.univariate.datasetcreator.DatasetCreator

NOTES:

TODO:
"""
from eqlearner.dataset.univariate.datasetcreator import DatasetCreator
# from eqlearner.dataset.univariate.equationtracker import EquationTracker
from EquationTrackerRG import EquationTrackerRG as EquationTracker
from EquationStructuresRG import EquationStructuresRG as EquationStructures
from EqGeneratorRG import constant_adder_binomial, polynomial_single, Composition_single, Binomial_single, N_single
import numpy as np


import copy


class DatasetCreatorRG(DatasetCreator):

    def number_of_terms(self):
        if self.random_terms:
            linear_terms = np.random.random_integers(0, self.max_linear_terms)
            binomial_terms = np.random.random_integers(0, self.max_binomial_terms)
            N_terms = np.random.random_integers(0, self.max_N_terms)
            compositions = np.random.random_integers(0, self.max_compositions)
            return linear_terms, binomial_terms, N_terms, compositions
        return self.max_linear_terms, self.max_binomial_terms, self.max_N_terms, self.max_compositions

    @staticmethod
    def handling_nan_evaluation(X, lambda_fun, X_noise=0, Y_noise=0):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("error")
            y = []
            for i in X:
                try:
                    y.append(lambda_fun(i))
                except RuntimeWarning:
                    y.append(np.nan)
        return y

    # def evaluate_function(self, X, sym_function, X_noise=0, Y_noise=0):
    #     """Difference is that ComplexInfinity issue is catch."""
    #     x = self.symbol
    #     try:
    #         function = lambdify(x, sym_function)
    #         # All the warnings are related to function not correctly evaluated. So we catch them and set a nan.
    #         y = self.handling_nan_evaluation(X, function, X_noise=X_noise, Y_noise=Y_noise)
    #         y = np.array(y)
    #         return y
    #     except KeyError:    # ComplexInfinity
    #         return np.array([np.nan])

    def generate_fun(self):
        """Difference from original is removal of set
        since this destroys repreducibility.

        Specifically, the line
            personalized_basis_fun = curr = set(random.choices(self.basis_functions_to_deprecate,k=2))
        became
            personalized_basis_fun = np.random.choice(self.basis_functions_to_deprecate, size=2, replace=False)
        """
        singles = []
        raw = []
        n_linear_terms, n_binomial_terms, n_N_terms, n_compositions = self.number_of_terms()
        total_combinations = EquationStructures(self.raw_basis_functions)
        tracker = EquationTracker(total_combinations.polynomial)
        for i in range(n_linear_terms):     # range(n_linear_terms):
            singles, raw = polynomial_single(tracker, singles, raw,
                                             self.symbol, constant_interval=self.interval_int)
        singles = self.order_fun(singles)
        singles_clean = [EquationStructures.polynomial_joiner(num, self.symbol) for num in raw]

        binomial = []
        priotity_list = []
        for i in range(0):  # range(n_binomial_terms):
            binomial, priotity_list = Binomial_single(total_combinations.binomial, binomial, priotity_list,
                                                      self.symbol, constant_interval=self.interval_int)
        binomial = self.order_fun(binomial)
        binomial_clean = [constant_adder_binomial(total_combinations.binomial[num], self.symbol) for num in priotity_list]

        N_terms = []
        priotity_list = []
        personalized_basis_fun = np.random.choice(self.basis_functions_to_deprecate, size=2, replace=False)
        for i in range(n_N_terms):
            N_terms, priotity_list = N_single(personalized_basis_fun, N_terms, priotity_list, 6)
        N_terms = self.order_fun(N_terms)
        N_terms_clean = copy.deepcopy(N_terms)

        compositions = []
        raw = []
        tracker = EquationTracker(total_combinations.compositions)

        for i in range(n_compositions):
            compositions, raw = Composition_single(tracker, compositions, raw,
                                                   self.symbol, constant_interval=self.interval_ext)

        compositions = self.order_fun(compositions)
        compositions_clean = [EquationStructures.composition_joiner(num, self.symbol) for num in raw]

        # division = []
        # priotity_list = []
        # if self.division_on:
        #     division, priotity_list = Division_single(self.basis_functions_to_deprecate,3)
        # division = self.order_fun(division)
        # division_clean = copy.deepcopy(division)

        dictionary = {"Single": singles, "binomial": binomial, "N_terms": N_terms, "compositions": compositions}    # , "division": division}
        res, dictionary = self.assembly_fun(dictionary)
        dictionary_cleaned = {"Single": singles_clean, "binomial": binomial_clean, "N_terms": N_terms_clean, "compositions": compositions_clean}    # , "division": division_clean}
        return res, dictionary, dictionary_cleaned  # singles, binomial_terms, N_terms, compositions, division)

    def assembly_fun(self, dictionary):
        res = 0
        for key, item in dictionary.items():
            for idx, elem in enumerate(dictionary[key]):
                c = random_from_intervals(self.interval_ext)
                dictionary[key][idx] = dictionary[key][idx] * c
                res = res + dictionary[key][idx]
        return res, dictionary


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
