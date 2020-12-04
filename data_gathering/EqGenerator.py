"""
AUTHOR: Ryan Grindle

PURPOSE: Generate dataset for meta symbolic regression
         infix notation.

NOTES:

TODO: Add constants
      Make rule generator more consistent S -> Mult(S,S) instead of
        S -> S*S even though for one arg it does S -> sin(S) for example.
      Spot and avoid obvious places that will result in duplicate equations
       (before removing duplicate function). Example: commutativity
"""
import sympy  # type: ignore
import numpy as np  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

import os
import json
from typing import List, Tuple, Dict


def insert_str(string, to_insert, index):
    return string[:index] + to_insert + string[index+1:]


class DatasetGenerator:
    """This class generates equations in infix notation
    using a grammar (although it is evaluated in an
    unusual way to create all possible equations)."""

    def __init__(self,
                 num_args: Dict[str, int],
                 max_depth: int,
                 X: np.ndarray) -> None:
        """
        PARAMETERS
        ----------
        num_args : dict
            The keys of this dictionary are the primitive
            set and the values are the number of arguments
            needed. (e.g. num_args['sin'] = 1)
        max_depth : int
            Max depth of tree associated with equation.
        tokens : List[str]
            A list of all the possible tokens.
        X : np.ndarray
            The x-values to use to compute the y-values.
        """
        assert all([1 <= n <= 2 for n in num_args.values()]), (
               'EqGenerator only supports primitive with one or two inputs.')
        self.max_depth = max_depth
        self.X = X

        self.tokens = list(num_args.keys())
        self.tokens += ['x', '(', ')', '^']
        self.tokens += [str(d) for d in range(10)]
        self.tokens += ['START', 'STOP']
        self.onehot_encoder = OneHotEncoder().fit([[t] for t in self.tokens])

        # Construct the rules from primitive set and num_args
        # All rules convert S to something (e.g. S -> (S+S))
        # so I use shorthand by not specifying the LHS.
        self.rules = ['x']
        for p in num_args:
            if num_args[p] == 1:
                self.rules.append('{}(S)'.format(p))
            else:
                self.rules.append('(S{}S)'.format(p))

    def gen_all_eqs(self, str_list: List[str] = ['S'],
                    depth: int = 0) -> List[str]:
        new_str_list = []
        all_eqs = []
        for s in str_list:
            next_level = self.iterate_eq_str(s)
            complete_eq, incomplete_eq = self.get_complete(next_level)
            new_str_list.extend(incomplete_eq)
            all_eqs.extend(complete_eq)
        if depth < self.max_depth:
            all_eqs.extend(self.gen_all_eqs(new_str_list, depth+1))
        return all_eqs

    def get_complete(self, str_list: List[str]) -> Tuple[List[str], List[str]]:
        incomplete = []
        complete = []
        for s in str_list:
            if 'S' in s:
                incomplete.append(s)
            else:
                complete.append(s)
        return complete, incomplete

    def iterate_eq_str(self, string: str,
                       S_count: int = None,
                       S_locs: List[int] = None) -> List[str]:
        if S_count is None:
            S_count = string.count('S')
        if 'S' not in string or S_count == 0:
            return [string]
        else:
            if S_locs is None:
                S_locs = [i for i, s in enumerate(string) if s == 'S']
            next_level = []
            for p in self.rules:
                new_string = string[:S_locs[-1]] + p + string[S_locs[-1]+1:]
                next_level.extend(self.iterate_eq_str(new_string,
                                                      S_count-1,
                                                      S_locs[:-1]))
            return next_level

    def format_expr_str(self, expr_str: str):
        return sympy.sympify(expr_str).expand()

    def remove_duplicates(self, eq_list: List[str]):
        eq_list = [self.format_expr_str(eq) for eq in eq_list]
        eq_list_no_coeff = [self.remove_coeff(eq) for eq in eq_list]
        _, indices = np.unique(eq_list_no_coeff, return_index=True)
        return [eq_list_no_coeff[i] for i in indices]

    def remove_const_mult_at(self, term: str, index: int) -> str:
        end_index = index
        while term[end_index].isdigit():
            end_index += 1
        if end_index == index:
            return term
        else:
            return term[:index] + term[end_index+1:]

    def remove_coeff_term(self, term: str) -> str:
        term = self.remove_const_mult_at(term, 0)
        paren_indices = [i+1 for i, t in enumerate(term) if t == '(']
        for i in paren_indices:
            term = self.remove_const_mult_at(term, i)
        return term

    def remove_coeff(self, expr) -> str:
        term_list = str(expr).split('+')
        no_coeff_terms = [self.remove_coeff_term(t.strip()) for t in term_list]
        return '+'.join(no_coeff_terms)

# def get_func_form(self, expr):
#     assert len(expr.args) <= 10
#     return '+'.join(['c{}*{}'.format(i, a) for i, a in enumerate(expr.args)])

    def get_dataset(self, eq_list) -> Tuple[np.ndarray]:
        self.dataset_input = []
        self.dataset_output = []
        for eq in eq_list:
            self.dataset_input.append(self.get_dataset_input(eq))
            self.dataset_output.append(self.get_dataset_output(eq))
        self.pad_output()
        self.dataset_output = [self.get_onehot(e) for e in self.dataset_output]

    def get_dataset_input(self, eq) -> np.ndarray:
        f = self.get_f(eq)
        return f(self.X).tolist()

    def get_f(self, expr):
        return np.vectorize(sympy.utilities.lambdify(sympy.Symbol('x'), expr))

    def get_dataset_output(self, eq):
        eq_str = str(eq).replace('**', '^')
        i = 0
        eq_seq = []
        while i < len(eq_str):
            for a in self.tokens:
                if eq_str[i:i+len(a)] == a:
                    eq_seq.append(eq_str[i:i+len(a)])
                    break
            i += len(a)
        return eq_seq

    def pad_output(self):
        max_len_output = max([len(eq_seq) for eq_seq in self.dataset_output])
        for eq_seq in self.dataset_output:
            eq_seq.extend(['STOP']*(max_len_output-len(eq_seq)))

    def get_onehot(self, seq):
        reshaped_seq = [[token] for token in seq]
        return self.onehot_encoder.transform(reshaped_seq).toarray().tolist()

    def save_dataset(self, save_name: str,
                     save_loc: str = os.path.join('..', 'datasets')) -> None:
        json.dump((self.dataset_input, self.dataset_output),
                  open(os.path.join(save_loc, save_name), 'w'),
                  separators=(',', ':'),
                  sort_keys=False,
                  indent=4)

    def load_dataset(self, save_name: str,
                     save_loc: str = os.path.join('..', 'datasets')):
        dataset_file = open(os.path.join(save_loc, save_name), 'r')
        dataset_inputs, dataset_outputs = json.load(dataset_file)
        return np.array(dataset_inputs), np.array(dataset_outputs)


if __name__ == '__main__':
    DG = DatasetGenerator(num_args={'*': 2, '+': 2, 'sin': 1},
                          max_depth=2,
                          X=np.linspace(0.1, 3.1, 30))
    eq_list = DG.gen_all_eqs()
    eq_list = DG.remove_duplicates(eq_list)
    print('eq_list', eq_list)
    print('len(eq_list)', len(eq_list))
    DG.get_dataset(eq_list)
    DG.save_dataset('dataset.json')

    dataset_input, dataset_output = DG.load_dataset('dataset.json')
    print('len(DG.tokens)', len(DG.tokens))
    print('dataset_input.shape', dataset_input.shape)
    print('dataset_output.shape', dataset_output.shape)
