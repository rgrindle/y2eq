from typing import Dict, Set

import numpy as np


def insert_str(string, to_insert, index):
    return string[:index] + to_insert + string[index+1:]


def get_locs_of(string, what):
    return [i for i, s in enumerate(string) if s == what]


class Grammar:

    def __init__(self, V: Set[str],
                 S: str, P: Dict[str, str],
                 rng=np.random.RandomState(0)):
        """
        PARAMETERS
        ----------
        V : set of str
            symbols with rules
        S : str
            start string
        P : dict from V to element(s) of union(V,U)
            rules (productions)
        """
        self.V = V
        self.S = S
        self.P = P
        self.rng = rng

    def gen_rand_str(self, string=None, depth=0, max_depth=7):
        string = self.S if string is None else string
        if not any([v in string for v in self.V]):
            return string
        else:
            new_string_list = []
            for s in string:
                if s in self.V:
                    if depth < max_depth-1:
                        sub_s = self.rng.choice(self.P[s])
                    else:
                        sub_s = 'x'
                    new_s = self.gen_rand_str(sub_s, depth=depth+1)
                else:
                    new_s = s
                new_string_list.append(new_s)
            return ''.join(new_string_list)

    def gen_all_list(self, str_list=['S'], depth=0, max_depth=7):
        new_str_list = []
        return_list = []
        for s in str_list:
            all_next_level = self.get_all_one_level(s)
            complete_eq, incomplete_eq = self.get_complete(all_next_level)
            new_str_list.extend(incomplete_eq)
            return_list.extend(complete_eq)
        if depth < max_depth:
            return_list.extend(self.gen_all_list(new_str_list, depth+1, max_depth))
        return return_list

    def get_complete(self, str_list):
        incomplete = []
        complete = []
        for s in str_list:
            if 'S' in s:
                incomplete.append(s)
            else:
                complete.append(s)
        return complete, incomplete

    def get_all_one_level(self, string, S_count=None, S_locs=None):
        S_count = string.count('S') if S_count is None else S_count
        if 'S' not in string or S_count == 0:
            return [string]
        else:
            S_locs = get_locs_of(string, 'S') if S_locs is None else S_locs
            next_level = []
            for p in self.P['S']:
                next_level.extend(self.get_all_one_level(insert_str(string, p, S_locs[-1]), S_count-1, S_locs[:-1]))
            return next_level


if __name__ == '__main__':
    G = Grammar(V={'S'},
                S='S', P={'S': ['x', '(S*S)']},#, '(S+S)',
                                # '(S-S)',
                                # 'sin(S)']},
                rng=np.random.RandomState(4))
    # next_level = G.get_all_one_level('x*(S*S)')
    # print(next_level)
    eq_list = G.gen_all_list(max_depth=2)
    print('eq_list', eq_list)

    # print(G.get_complete(['x*(x*x)', 'x*((S*S)*x)', 'x*(x*(S*S))', 'x*((S*S)*(S*S))']))
    # import sympy
    # x = sympy.symbols('x')
    # def format_expr_str(expr_str):
    #     return sympy.sympify(string).expand()
    # expr = format_expr_str(string)
    # print(expr)
