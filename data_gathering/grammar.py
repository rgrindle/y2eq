from typing import Dict, Set

import numpy as np


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


if __name__ == '__main__':
    G = Grammar(V={'S'},
                S='S', P={'S': ['x', '(S*S)', '(S+S)',
                                '(S-S)',
                                'sin(S)']},
                rng=np.random.RandomState(4))
    string = G.gen_rand_str()
    print(string)
    import sympy
    x = sympy.symbols('x')
    expr = sympy.sympify(string)
    print(expr)
