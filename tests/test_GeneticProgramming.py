from gp.GeneticProgramming import GeneticProgramming
from gp.Individual import Individual
from gp.RegressionDataset import RegressionDataset
from check_assert import check_assert

import numpy as np

import os


def setup_GP():

    train_x = RegressionDataset.linspace(-1, 1, 20)
    val_x = RegressionDataset.linspace(0, 1, 20)
    test_x = RegressionDataset.linspace(-2, 2, 20)

    train_dataset = RegressionDataset(x=train_x, f=lambda x: x[0])
    val_dataset = RegressionDataset(x=val_x, f=lambda x: x[0])
    test_dataset = RegressionDataset(x=test_x, f=lambda x: x[0])

    GP = GeneticProgramming(rep=0, exp=0, taskid=0,
                            dataset_type='train',
                            pop_size=12, max_gens=10,
                            primitive_set=['*', '+', '-'],
                            terminal_set=['x0'],
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            test_dataset=test_dataset)

    return GP


def setup_individual(tree):

    tree = Individual(rng=np.random.RandomState(0),
                      tree=tree,
                      primitive_set=['*', '+', '-'],
                      terminal_set=['#x'])

    return tree


def test_generate_population_ramped_half_and_half():

    GP = setup_GP()
    pop = GP.generate_population_ramped_half_and_half(GP.pop_size, 6)
    yield check_assert, len(pop) == GP.pop_size

    depths = [p.get_depth() for p in pop]
    yield check_assert, np.all([i in depths for i in range(1, 6)])


def test_compute_fitness():

    GP = setup_GP()
    ind1 = setup_individual('(x0)')
    ind2 = setup_individual('(- (x0) (x0))')
    half_pop = GP.pop_size//2
    GP.pop = [ind1]*half_pop + [ind2]*half_pop
    GP.compute_fitness()
    fitnesses = [p.fitness for p in GP.pop]
    val_fitnesses = [p.validation_fitness for p in GP.pop]

    yield check_assert, len(np.unique(fitnesses[:half_pop])) == 1
    yield check_assert, len(np.unique(fitnesses[half_pop:])) == 1
    yield check_assert, len(np.unique(val_fitnesses[:half_pop])) == 1
    yield check_assert, len(np.unique(val_fitnesses[half_pop:])) == 1

    yield check_assert, np.array(fitnesses[:half_pop]) == 0.


def test_get_summary_info():

    GP = setup_GP()
    GP.compute_fitness()
    info = GP.get_summary_info()

    for key in ['train_fitness', 'val_fitness', 'tree_size', 'tree_depth']:
        yield check_assert, key in info

        for subkey in ['min', 'max', 'mean', 'median']:
            yield check_assert, subkey in info[key]
            yield check_assert, type(info[key][subkey]) in (np.int64, np.float64)


def test_run_generation():

    GP = setup_GP()
    GP.compute_fitness()
    pop_before = GP.pop
    GP.run_generation(gen=1,
                      output_path='',
                      num_to_mutate=GP.pop_size)
    pop_after = GP.pop

    yield check_assert, len(pop_before) == len(pop_after)
    yield check_assert, [p.fitness for p in pop_before] != [p.fitness for p in pop_after]


def test_tournament_selection():

    GP = setup_GP()
    ind1 = setup_individual('(x0)')
    ind2 = setup_individual('(- (x0) (x0))')
    GP.pop = [ind1, ind2]
    GP.pop_size = 2
    GP.compute_fitness()

    best_individual = GP.tournament_selection(2)

    yield check_assert, best_individual.get_lisp_string() == '(x0)'


def test_update_best_individual():

    GP = setup_GP()
    fake_fitnesses = list(range(GP.pop_size))
    np.random.shuffle(fake_fitnesses)

    for i, f in enumerate(fake_fitnesses):
        GP.pop[i].validation_fitness = f
        GP.pop[i].fitness = i

    GP.update_best_individual()

    yield check_assert, GP.best_individual.validation_fitness == 0
    yield check_assert, GP.best_individual.fitness == fake_fitnesses.index(0)


def test_run():

    GP = setup_GP()
    GP.run('.')

    summary_file = 'all_summary_exp0_rep0_taskid0_datasettypetrain.csv'
    best_file = 'best_exp0_rep0_taskid0_datasettypetrain.csv'

    yield check_assert, os.path.exists(summary_file), 'history of summary data does not exist'
    yield check_assert, os.path.exists(best_file), 'history of best does not exist'

    # clean up (delete file)
    os.remove(summary_file)
    os.remove(best_file)


def test_stop():

    GP = setup_GP()
    GP.max_gens = 10

    gen_list = [9, 10, 11]
    answer_list = [False, False, True]

    for gen, answer in zip(gen_list, answer_list):
        yield check_assert, GP.stop(gen) == answer
