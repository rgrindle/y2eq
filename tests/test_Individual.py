from gp.Individual import Individual
from gp.RegressionDataset import RegressionDataset
from check_assert import check_assert

import numpy as np

import copy


def setup_individual():

    ind = Individual(rng=np.random.RandomState(0),
                     primitive_set=['*', '+', '-'],
                     terminal_set=['#x'])

    return ind


def test_generate_individual_full():

    ind = setup_individual()

    new_ind = ind.generate_individual_full(0)
    yield check_assert, new_ind == ['#x']

    for max_depth in range(10):
        new_ind = Individual(tree=ind.generate_individual_full(max_depth),
                             primitive_set=ind.P,
                             terminal_set=ind.T,
                             rng=np.random.RandomState(0))
        yield check_assert, new_ind.get_depth() == max_depth


def test_generate_individual_grow():

    ind = setup_individual()

    new_ind = ind.generate_individual_grow(0)
    yield check_assert, new_ind == ['#x']

    for max_depth in range(10):
        new_ind = Individual(tree=ind.generate_individual_grow(max_depth),
                             primitive_set=ind.P,
                             terminal_set=ind.T,
                             rng=np.random.RandomState(0))

        yield check_assert, new_ind.get_depth() <= max_depth


def test_mutate():

    ind = setup_individual()
    new_ind = ind.mutate(max_node_growth=2)

    yield check_assert, ind.get_depth() + 2 >= new_ind.get_depth()


def test_node_replacement():

    ind = setup_individual()

    ind.node_replacement(subtree=ind,
                         child_indices=(),
                         max_node_growth=2)

    ind.from_string('(+ (x0) (x0))')
    ind_copy = copy.deepcopy(ind)
    for _ in range(3):
        ind.rng.rand()
    ind.node_replacement(subtree=ind,
                         child_indices=(1,),
                         max_node_growth=2)
    yield check_assert, 1 <= ind.get_depth() == 3

    node_dict_unmutated = ind_copy.get_node_dict()
    node_dict_mutated = ind.get_node_dict()

    for loc in node_dict_mutated:
        # if not in mutated part of tree
        if (1,) != loc[:1]:
            yield check_assert, node_dict_mutated[loc] == node_dict_unmutated[loc]


def test_evaluate():

    ind = setup_individual()
    ind.from_string('(* (x0) (x0))')

    x = RegressionDataset.linspace(-1, 1, 5)
    f = lambda x: 2*x[0]
    train_dataset = RegressionDataset(x=x, f=f)

    ind.evaluate(train_dataset)
    yield check_assert, np.round(ind.fitness, 8) == 1.55724115

    val_dataset = RegressionDataset(x=x, f=lambda x: x[0])
    ind.evaluate(train_dataset, val_dataset)
    yield check_assert, np.round(ind.validation_fitness, 10) == 0.9617692031


def test_evaluate_test_points():

    ind = setup_individual()
    ind.from_string('(* (x0) (x0))')

    x = RegressionDataset.linspace(-1, 1, 5)
    train_dataset = RegressionDataset(x=x, f=lambda x: 2*x[0])
    ind.evaluate(train_dataset)

    test_dataset = RegressionDataset(x=x, f=lambda x: 0*x[0])
    ind.evaluate_test_points(test_dataset)

    yield check_assert, np.round(ind.testing_fitness, 10) == 0.6519202405
