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
from Equation import Equation

import numpy as np  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

import os
import json
from typing import List, Tuple, Dict

np.seterr(all='raise')


def insert_str(string, to_insert, index):
    return string[:index] + to_insert + string[index+1:]


class DatasetGenerator:
    """This class generates equations in infix notation
    using a grammar (although it is evaluated in an
    unusual way to create all possible equations)."""

    def __init__(self,
                 num_args: Dict[str, int],
                 max_depth: int,
                 X: np.ndarray,
                 rng,
                 include_zero_eq: bool = False,
                 load_eq_save_name: str = None,
                 num_coeff_sets: int = 1) -> None:
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
        self.rng = rng
        self.include_zero_eq = include_zero_eq
        self.num_coeff_sets = num_coeff_sets

        self.tokens = list(num_args.keys())
        self.tokens += ['x0', '(', ')', '^']
        self.tokens += [str(d) for d in range(10)]
        self.tokens += ['START', 'STOP']
        self.onehot_encoder = OneHotEncoder().fit([[t] for t in self.tokens])

        # Construct the rules from primitive set and num_args
        # All rules convert S to something (e.g. S -> (S+S))
        # so I use shorthand by not specifying the LHS.
        self.rules = ['x0']
        for p in num_args:
            if num_args[p] == 1:
                self.rules.append('{}(S)'.format(p))
            else:
                self.rules.append('(S{}S)'.format(p))

        if load_eq_save_name is None:
            print('Generating all equations '
                  '(max_depth={}) ...'.format(self.max_depth))
            self.gen_all_eqs()
            print('Found', len(self.all_eqs), 'equations')
            print('Reducing to unique equation ...')
            self.all_eqs = np.unique(self.all_eqs).tolist()
            print('Found', len(self.all_eqs), 'unique equations')
        else:
            self.load_eqs(load_eq_save_name)

    def gen_all_eqs(self):
        self.all_eqs = [Equation(eq) for eq in self.__gen_all_eqs()]
        if self.include_zero_eq:
            self.all_eqs.insert(0, Equation('0'))
        return self.all_eqs

    def __gen_all_eqs(self, str_list: List[str] = ['S'],
                      depth: int = 0) -> List[str]:
        new_str_list = []
        all_eqs = []
        for s in str_list:
            next_level = self.iterate_eq_str(s)
            complete_eq, incomplete_eq = self.get_complete(next_level)
            new_str_list.extend(incomplete_eq)
            all_eqs.extend(complete_eq)
        if depth < self.max_depth:
            all_eqs.extend(self.__gen_all_eqs(new_str_list, depth+1))
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

    def get_dataset(self) -> Tuple[np.ndarray]:
        self.dataset_input = []
        self.non_dataset_eqs = []
        self.dataset_eqs = []
        for eq in self.all_eqs:
            try:
                for _ in range(self.num_coeff_sets):
                    Y = self.get_Y(eq)
                    self.dataset_input.append(Y)
                    self.get_eq_seq(eq)
                    self.dataset_eqs.append(eq)
            except FloatingPointError:
                self.non_dataset_eqs.append(eq)
                continue
        self.pad_eq_seqs()
        self.dataset_output = [self.get_onehot(e) for e in self.dataset_eqs]

    def pad_eq_seqs(self):
        max_len_output = max([len(eq.eq_seq) for eq in self.dataset_eqs])
        for eq in self.dataset_eqs:
            eq.eq_seq.extend(['STOP']*(max_len_output-len(eq.eq_seq)))

    def get_Y(self, eq) -> np.ndarray:
        iter_count = 0
        while True:
            try:
                eq.coeffs = self.rng.uniform(-5, 5, eq.num_coeffs)
                return eq.eval(self.X)
            except FloatingPointError:
                iter_count += 1
                if iter_count > 100:
                    raise FloatingPointError

    def get_eq_seq(self, eq):
        i = 0
        eq.eq_seq = []
        while i < len(eq.eq_str):
            for a in self.tokens:
                if eq.eq_str[i:i+len(a)] == a:
                    eq.eq_seq.append(eq.eq_str[i:i+len(a)])
                    break
            i += len(a)
        return eq.eq_seq

    def get_onehot(self, eq):
        __seq = [[token] for token in eq.eq_seq]
        eq.onehot = self.onehot_encoder.transform(__seq).toarray().tolist()
        return eq.onehot

    def save_dataset(self, save_name: str,
                     save_loc: str = os.path.join('..', 'datasets')) -> None:
        json.dump((self.dataset_input, self.dataset_output),
                  open(os.path.join(save_loc, save_name), 'w'),
                  separators=(',', ':'),
                  sort_keys=False,
                  indent=4)

    def save_eqs(self, save_name: str,
                 save_loc: str = os.path.join('..', 'datasets')) -> None:
        json.dump([str(eq) for eq in self.all_eqs],
                  open(os.path.join(save_loc, save_name), 'w'),
                  separators=(',', ':'),
                  sort_keys=False,
                  indent=4)

    def load_dataset(self, save_name: str,
                     save_loc: str = os.path.join('..', 'datasets')):
        dataset_file = open(os.path.join(save_loc, save_name), 'r')
        dataset_inputs, dataset_outputs = json.load(dataset_file)
        return np.array(dataset_inputs), np.array(dataset_outputs)

    def load_eqs(self, save_name: str,
                 save_loc: str = os.path.join('..', 'datasets')):
        dataset_file = open(os.path.join(save_loc, save_name), 'r')
        eq_strs = json.load(dataset_file)
        self.all_eqs = [Equation(eq.replace('^', '**')) for eq in eq_strs]


if __name__ == '__main__':

    def yes_or_no(message):
        answer = None
        while answer not in ('yes', 'no'):
            answer = input(message+' (yes or no) ')
        return answer == 'yes'

    MAX_DEPTH = 3
    SEED = 1
    unique_eqs_save_file = 'unique_eqs_maxdepth{}.json'.format(MAX_DEPTH)
    dataset_save_file = 'dataset_maxdepth{}_seed{}.json'.format(MAX_DEPTH, SEED)
    save_path = os.path.join('..', 'datasets')

    if os.path.isfile(os.path.join(save_path, unique_eqs_save_file)):
        print('Found: ', unique_eqs_save_file)
        use_file = yes_or_no('Use this file to create dataset?')
    else:
        use_file = False

    load_eq_save_name = unique_eqs_save_file if use_file else None

    DG = DatasetGenerator(num_args={'*': 2, '+': 2, 'sin': 1,
                                    'log': 1, 'exp': 1},
                          max_depth=MAX_DEPTH,
                          X=np.linspace(0.1, 3.1, 30),
                          rng=np.random.RandomState(SEED),
                          include_zero_eq=True,
                          load_eq_save_name=load_eq_save_name,
                          num_coeff_sets=1)

    if not use_file:
        print('Saving unique equations ...')
        DG.save_eqs(unique_eqs_save_file)
        print('Unique equations saved:', unique_eqs_save_file)

    print('Formatting unique equations into dataset ...')
    DG.get_dataset()
    print('Dataset has', len(DG.dataset_eqs), 'observations')

    print('Saving dataset ...')
    DG.save_dataset(dataset_save_file)
    print('Dataset saved:', dataset_save_file)

    if len(DG.non_dataset_eqs) > 0:
        print('Equations excluded from datatset '
              'because suitable constants could not '
              'be found are listed below.')
        for eq in DG.non_dataset_eqs:
            print(eq)
