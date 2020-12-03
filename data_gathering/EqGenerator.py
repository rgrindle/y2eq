"""
AUTHOR: Ryan Grindle

PURPOSE: Generate dataset for meta symbolic regression
         infix notation.

NOTES:

TODO: Make consistent format and remove duplicates
      Add constants
      Calculate semantics
      Form the dataset (maybe a different class that uses this one?)
      Make rule generator more consistent S -> Mult(S,S) instead of
        S -> S*S even though for one arg it does S -> sin(S) for example.
"""

from typing import Set, List, Tuple, Dict


def insert_str(string, to_insert, index):
    return string[:index] + to_insert + string[index+1:]


class EqGenerator:
    """This class generates equations in infix notation
    using a grammar (although it is evaluated in an
    unusual way to create all possible equations)."""

    def __init__(self,
                 primitive_set: Set[str],
                 num_args: Dict[str, int],
                 max_depth: int) -> None:
        """
        PARAMETERS
        ----------
        primitive_set : set
            primitive set to use in generation
            of equations.
        num_args : dict
            How many arguments are expected for each
            element of the primitive set? (e.g. num_args['sin'] = 1)
        max_depth : int
            Max depth of tree associated with equation.
        """
        assert primitive_set == num_args.keys()
        assert all([1 <= n <= 2 for n in num_args.values()])
        self.primitive_set = primitive_set
        self.num_args = num_args
        self.max_depth = max_depth

        # Construct the rules from primitive set and num_args
        # All rules convert S to something (e.g. S -> (S+S))
        # so I use shorthand by not specifying the LHS.
        self.rules = ['x']
        for p in self.primitive_set:
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


if __name__ == '__main__':
    G = EqGenerator(primitive_set={'*', '+'},
                    num_args={'*': 2, '+': 2},
                    max_depth=2)
    eq_list = G.gen_all_eqs()
    print('eq_list', eq_list)
