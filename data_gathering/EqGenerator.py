"""
AUTHOR: Ryan Grindle

PURPOSE: Generate dataset for meta symbolic regression
         infix notation.

NOTES:

TODO: Make this more ituitive and constuct P from primitive set.
      Make consistent format and remove duplicates
      Add constants
      Calculate semantics
      Form the dataset (maybe a different class that uses this one?)
"""

from typing import Dict


def insert_str(string, to_insert, index):
    return string[:index] + to_insert + string[index+1:]


class EqGenerator:

    def __init__(self,
                 P: Dict[str, str],
                 max_depth):
        """
        PARAMETERS
        ----------
        P : dict
            rules (productions)
        """
        self.P = P
        self.max_depth = max_depth

    def gen_all_eqs(self, str_list=['S'],
                    depth=0):
        new_str_list = []
        all_eqs = []
        for s in str_list:
            all_next_level = self.iterate_eq_str(s)
            complete_eq, incomplete_eq = self.get_complete(all_next_level)
            new_str_list.extend(incomplete_eq)
            all_eqs.extend(complete_eq)
        if depth < self.max_depth:
            all_eqs.extend(self.gen_all_eqs(new_str_list, depth+1))
        return all_eqs

    def get_complete(self, str_list):
        incomplete = []
        complete = []
        for s in str_list:
            if 'S' in s:
                incomplete.append(s)
            else:
                complete.append(s)
        return complete, incomplete

    def iterate_eq_str(self, string, S_count=None, S_locs=None):
        if S_count is None:
            S_count = string.count('S')
        if 'S' not in string or S_count == 0:
            return [string]
        else:
            if S_locs is None:
                S_locs = [i for i, s in enumerate(string) if s == 'S']
            next_level = []
            for p in self.P['S']:
                new_string = string[:S_locs[-1]] + p + string[S_locs[-1]+1:]
                next_level.extend(self.iterate_eq_str(new_string,
                                                      S_count-1,
                                                      S_locs[:-1]))
            return next_level


if __name__ == '__main__':
    G = EqGenerator(P={'S': ['x', '(S*S)', '(S+S)']},  # '(S-S)', 'sin(S)']},
                    max_depth=2)
    eq_list = G.gen_all_eqs()
    print('eq_list', eq_list)
