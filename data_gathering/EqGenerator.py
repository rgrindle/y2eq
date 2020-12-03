"""
AUTHOR: Ryan Grindle

PURPOSE: Generate dataset for meta symbolic regression
         infix notation.

NOTES:

TODO: Add constants
      Calculate semantics
      Form the dataset (maybe a different class that uses this one?)
      Make rule generator more consistent S -> Mult(S,S) instead of
        S -> S*S even though for one arg it does S -> sin(S) for example.
      Spot and avoid obvious places that will result in duplicate equations
       (before removing duplicate function). Example: commutativity
"""
import sympy
import numpy as np

from typing import List, Tuple, Dict


def insert_str(string, to_insert, index):
    return string[:index] + to_insert + string[index+1:]


class EqGenerator:
    """This class generates equations in infix notation
    using a grammar (although it is evaluated in an
    unusual way to create all possible equations)."""

    def __init__(self,
                 num_args: Dict[str, int],
                 max_depth: int) -> None:
        """
        PARAMETERS
        ----------
        num_args : dict
            The keys of this dictionary are the primitive
            set and the values are the number of arguments
            needed. (e.g. num_args['sin'] = 1)
        max_depth : int
            Max depth of tree associated with equation.
        """
        assert all([1 <= n <= 2 for n in num_args.values()]), (
               'EqGenerator only supports primitive with one or two inputs.')
        self.max_depth = max_depth

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

# def get_func_form(self, expr):
#     assert len(expr.args) <= 10
#     return '+'.join(['c{}*{}'.format(i, a) for i, a in enumerate(expr.args)])

    def format_expr_str(self, expr_str: str):
        return sympy.sympify(expr_str).expand()

    def remove_duplicates(self, eq_list: List[str]):
        eq_list = [self.format_expr_str(eq) for eq in eq_list]
        eq_list_no_coeff = [self.remove_coeff(eq) for eq in eq_list]
        _, indices = np.unique(eq_list_no_coeff, return_index=True)
        return [eq_list_no_coeff[i] for i in indices]

    def remove_coeff_term(self, term: str) -> str:
        if term[0].isdigit():
            return term[2:]
        else:
            return term

    def remove_coeff(self, expr):
        term_list = str(expr).split('+')
        no_coeff_terms = [self.remove_coeff_term(t.strip()) for t in term_list]
        return '+'.join(no_coeff_terms)


if __name__ == '__main__':
    G = EqGenerator(num_args={'*': 2, '+': 2},
                    max_depth=2)
    eq_list = G.gen_all_eqs()
    print('eq_list', eq_list)
    print(len(eq_list))
    eq_list = G.remove_duplicates(eq_list)
    print('eq_list', eq_list)
    print(len(eq_list))
