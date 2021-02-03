"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: The Individual class expands on the Tree class
         by giving the ability to generate an tree/equation
         mutate the tree/equation and evaluate the equation.
         This class is used by GeneticProgramming.

NOTES:

TODO:
"""
from gp.Tree import Tree
from gp.primitive_info import num_children
from gp.RegressionDataset import RegressionDataset
from equation.EquationLisp import EquationLisp

import numpy as np  # noqa: F401

import copy
from typing import List, Tuple


# def get_function_from_function_str(function_str: str) -> Callable[[Any], Any]:
#     if 'x' not in function_str:
#         function_str += '+x[0]*0'
#     inputs = 'x,c' if 'c[' in function_str else 'x'

#     return eval('lambda '+inputs+': '+function_str)


class Individual(Tree):
    """This class is a individual in a symbolic
    regression algorithm. It inherits the Tree
    class. In this class, fitness and error are
    synonimous."""

    def __init__(self, rng, primitive_set: List[str],
                 terminal_set: List[str], num_vars: int = 1,
                 max_depth: int = 6, depth: int = None,
                 tree=None, method: str = None, **params):
        """Initialize Individual

        Parameters
        ----------
        rng : random number generator
            For example let rng=np.random.RandomState(0)
        primitive_set : list
            A list of all primitive (operators/functions)
            that may be used in trees.
        terminal_set: list
            A list of all allowed terminals (constants, variables).
        num_vars : int (default=1)
            The number of input variables to use. This must be
            specified if more than one input variable is necessary.
        max_depth : int (default=6)
            A non-negative integer that limits the depth of the
            tree.
        depth : int (optional)
            A non-negative integer that determines the initial
            max depth of the tree. This parameter should only
            be used in conjunction with argument called method.
            Otherwise, it is not used or assumed to be max_depth.
        tree : list (of lists, optional)
            Pass a list of list to represent the tree
            like Tree.tree.
        method : str
            A string such as 'grow' or 'full' which determines
            the method for creating the tree. If left blank,
            one of the previously mentioned methods is selected
            at random.
        """

        # Run parent classes __init__, but possibly don't
        # specify the tree yet
        Tree.__init__(self,
                      tree=tree,
                      num_vars=num_vars,
                      rng=rng,
                      **params)

        self.fitness = None

        # fitness (that does not effect other fitness)
        # during symbolic regression
        self.validation_fitness = 0

        # fitness on data after symbolic regression finishes
        self.testing_fitness = 0

        # We don't want any repeated elements in these lists,
        # but using rng.choice on set does not always produce
        # the same results, even with the same seed. So, we
        # use lists
        self.P = primitive_set
        self.T = terminal_set

        self.max_depth = max_depth

        if depth is None:
            depth = max_depth

        # Figure out which method to use to create tree.
        # If none specified pick one at random.
        if tree is not None:
            # already did this part by calling
            # parent __init__() above
            pass

        else:

            if method is None:
                # If full is false, use grow method
                full = self.rng.choice((False, True))

                if full:
                    self.tree = self.generate_individual_full(depth)

                else:
                    self.tree = self.generate_individual_grow(depth)

            elif method == 'grow':
                self.tree = self.generate_individual_grow(depth)

            elif method == 'full':
                self.tree = self.generate_individual_full(depth)

            else:
                print("Error:", method, "is an unknown method.")

            self.apply_rules_to_tree()

# ---------------------------------------------------- #
#               Tree Creation Functions
# ---------------------------------------------------- #

    def generate_individual_full(self, max_depth: int,
                                 depth: int = 0, subtree=None):

        if subtree is None:
            subtree = []

        if max_depth == depth:
            return [self.rng.choice(self.T)]

        else:
            primitive = self.rng.choice(self.P)

            children = []
            for i in range(num_children[primitive]):
                child = self.generate_individual_full(max_depth,
                                                      depth=depth+1,
                                                      subtree=None)
                children.append(child)

            subtree.extend([primitive] + children)

            return subtree

    def generate_individual_grow(self, max_depth: int,
                                 depth: int = 0, subtree=None):

        if subtree is None:
            subtree = []

        if max_depth == depth:
            return [self.rng.choice(self.T)]

        else:
            prim_or_term = self.rng.choice(self.P+self.T)

            if prim_or_term in self.P:
                children = []
                for i in range(num_children[prim_or_term]):
                    child = self.generate_individual_grow(max_depth,
                                                          depth=depth+1,
                                                          subtree=None)
                    children.append(child)

                subtree.extend([prim_or_term] + children)

            else:
                subtree.append(prim_or_term)

            return subtree

# --------------------------------------------------- #
#                      Mutations
# --------------------------------------------------- #

    def mutate(self, max_node_growth: int):
        """Pick a random node in the tree and then pick a mutation
        based on that particular node.

        Parameters
        ----------
        max_node_growth : int
            Mutation parameter describing the max_depth of subtree
            to create on mutation (node_replacement).

        Returns
        -------
        mutated_ind : Individual
            The mutated version of self.
        """

        # Get list of all node in tree (individual).
        node_list = self.get_node_list()

        # Choose one node for the mutation location
        index = self.rng.choice(len(node_list))
        child_indices = node_list[index]

        # Make a new tree that is currently identical to the old one.
        new_tree = self.__class__(rng=self.rng, primitive_set=self.P,
                                  terminal_set=self.T, num_vars=self.num_vars,
                                  max_depth=self.max_depth,
                                  tree=copy.deepcopy(self.tree), **self.params)

        # Mutate the individual.
        mutated_ind = self.node_replacement(new_tree, child_indices, max_node_growth)
        mutated_ind.apply_rules_to_tree()

        return mutated_ind

    def node_replacement(self, subtree,
                         child_indices: Tuple[int],
                         max_node_growth: int = 6):
        """Create new subtree and place it at choice_list.

        Parameters
        ----------
        subtree : list (of lists)
            The subtree in self.tree to be replaced.
        child_indices : iterable
            List of child indices that describes the location of node
            in tree.
        max_node_growth : int (default=6)
            Specifies the max depth of the subtree.

        Returns
        -------
        subtree : list (of lists)
            The subtree in self.tree to be replaced. It is now mutated.
        """

        if child_indices == [] or child_indices == ():
            depth = 0

        else:
            depth = len(child_indices)

        new_subtree = self.generate_individual_grow(min(max_node_growth, self.max_depth - depth))
        subtree.set_subtree(new_subtree, child_indices)

        return subtree

    # --------------------------------------------------------------- #
    #                          Compute Error
    # --------------------------------------------------------------- #

    def evaluate(self, train_dataset: RegressionDataset,
                 val_dataset: RegressionDataset = None) -> None:
        """Calculate error for training data and validation if
        provided.
        """
        f_string = self.convert_lisp_to_standard().replace('[0]', '')
        eq = EquationLisp(f_string,
                          x=train_dataset.x.flatten())
        eq.fit(train_dataset.y.flatten())
        self.fitness = eq.rmse
        self.coeffs = eq.coeffs
        self.f = lambda x: eq.f(c=eq.coeffs, x=x.flatten())

        if val_dataset is not None:
            self.validation_fitness = val_dataset.get_RMSE(self.f)

    def evaluate_test_points(self, test_dataset: RegressionDataset) -> None:
        """Calculate error for testing data. This method assumes that self.f
        exists, which is usually created by running
        Individual.evaluate_fitness
        """
        self.testing_fitness = test_dataset.get_RMSE(self.f)
