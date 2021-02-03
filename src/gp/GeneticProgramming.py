"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Sept 11, 2020

PURPOSE: The GeneticProgramming class performs symbolic regression
         by using the standard genetic programming algorithm.

NOTES:

TODO:
"""
from gp.Individual import Individual
from gp.RegressionDataset import RegressionDataset

import numpy as np
import pandas as pd

import os
import time
import copy
from typing import List, Dict, Any


class GeneticProgramming:
    """This class is used to create a population of Individuals
    and evolve them to solve a particular dataset"""

    def __init__(self, exp: int, rep: int, pop_size: int,
                 dataset_index: int,
                 primitive_set: List[str], terminal_set: List[str],
                 train_dataset: RegressionDataset, val_dataset: RegressionDataset,
                 test_dataset: RegressionDataset, num_vars: int = 1,
                 init_max_depth: int = 6, max_depth: int = 17,
                 individual=Individual, max_gens: int = float('inf'),
                 max_node_growth: int = 3, **individual_params) -> None:
        """Initialize GeneticProgramming

        Parameters
        ----------
        rng : random number generator
            For example let rng=np.random.RandomState(0)
        pop_size : int
            Number of individuals to put in the population.
        max_gens : int
            Maximum number of generations
        primitive_set : list
            A list of all primitive (operators/functions)
            that may be used in trees.
        terminal_set: list
            A list of all allowed terminals (constants, variables).
        test_data : np.array
            A 2D np.arrays. A row of data is of the form
            y, x0, x1, ...
        num_vars : int (default=1)
            The number of input variables to use. This must be
            specified if more than one input variable is necessary.
        init_max_depth : int (default=6)
            A non-negative integer that limits the depth of the
            trees initially created.
        max_depth : int (default=17)
            A non-negative integer that limits the depth of the
            tree.
        individual : Individual (or superclass)
            The version of the Individual class to use.
        max_node_growth : int
            Mutation parameter describing the max depth of subtree
            to be created on mutation.
        """

        self.rep = rep
        self.exp = exp
        self.rng = np.random.RandomState(100*self.exp+self.rep)

        self.dataset_index = dataset_index
        self.max_depth = max_depth
        self.init_max_depth = init_max_depth
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.P = primitive_set
        self.T = terminal_set
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.Individual = individual
        self.params = individual_params
        self.num_vars = num_vars
        self.max_node_growth = max_node_growth

        self.pop = self.generate_population_ramped_half_and_half(self.pop_size, self.init_max_depth)

        if 'save_pop_data' not in self.params:
            self.save_pop_data = False
        else:
            self.save_pop_data = self.params['save_pop_data']

        if 'timeout' in self.params:
            self.timeout = self.params['timeout']
        else:
            self.timeout = float('inf')

        self.start_time = time.process_time()

        # This is the best individual based
        # on validation error.
        self.best_individual = None

    def generate_population_ramped_half_and_half(self, size: int,
                                                 init_max_depth: int) -> List[Individual]:
        """Generate the population using the ramped half and half method.
        Generate equal number of individuals with full and grow method and
        an equal number of each of those with depth of size
        1, 2, 3, ... max_depth.

        Parameters
        ----------
        size : int
            Desired population size
        init_max_depth : int
            Max depth to use when generating trees.
        """

        new_pop = []

        group_size = int(size / (init_max_depth))
        half_group_size = int(group_size / 2)

        # make half the individual with the grow method and the
        # other half with the full method
        # increase max depth as we go
        for d in range(1, init_max_depth+1):
            for i in range(half_group_size):
                new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                               depth=d, method='full', max_depth=self.max_depth,
                                               **self.params))
                new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                               depth=d, method='grow', max_depth=self.max_depth,
                                               **self.params))

            # if group size doesn't divide easily make another
            # individual with either grow or full
            if group_size % 2 != 0:
                new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                               depth=d, method='full', max_depth=self.max_depth,
                                               **self.params))

        # if population size is not divisible by group_size
        i = len(new_pop)

        while len(new_pop) < size:
            new_pop.append(self.Individual(self.rng, self.P, self.T, num_vars=self.num_vars,
                                           depth=(i % init_max_depth)+2, max_depth=self.max_depth,
                                           **self.params))
            i += 1

        return new_pop

    def compute_fitness(self) -> None:
        """Compute fitness (error) for all individuals."""

        for i, individual in enumerate(self.pop):
            individual.evaluate(self.train_dataset, self.val_dataset)

    def get_summary_info(self) -> Dict[str, Dict[str, float]]:
        """Get the summary of the population. This includes:
        max, min, median, and mean of fitness objective,
        number of nodes in trees, depth of trees, and
        validation error.
        """

        train_fitnesses = [p.fitness for p in self.pop]
        val_fitnesses = [p.validation_fitness for p in self.pop]
        tree_sizes = [p.get_tree_size() for p in self.pop]
        tree_depths = [p.get_depth() for p in self.pop]

        return {'train_fitness': self.get_summary(train_fitnesses),
                'val_fitness': self.get_summary(val_fitnesses),
                'tree_size': self.get_summary(tree_sizes),
                'tree_depth': self.get_summary(tree_depths)}

    def get_summary(self, data: List[float]) -> Dict[str, float]:

        return {'min': np.min(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'max': np.max(data)}

    def run_generation(self, gen: int,
                       output_path: str,
                       num_to_mutate: int) -> None:
        """Stuff to repeat every generation.

        Parameters
        ----------
        gen : int
            The current generation.
        num_to_mutate: int
            The number of individual to mutate.
        """

        mut_parents = [self.tournament_selection(2) for _ in range(num_to_mutate)]
        new_pop = []

        # Go through all individuals and edit the population.
        for parent in mut_parents:
            new_ind = parent.mutate(self.max_node_growth)
            new_pop.append(new_ind)

        # use elitism if there are not enough mutants
        num_missing = self.pop_size - len(new_pop)
        if num_missing != 0:
            fitnesses = [p.fitness for p in self.pop]
            elite_indices = np.argsort(fitnesses)[:num_missing]

            for i in elite_indices:
                new_pop.append(copy.deepcopy(self.pop[i]))

        self.pop = new_pop

        # Evaluate the entire population.
        self.compute_fitness()

    def tournament_selection(self, group_size):

        indices = self.rng.choice(self.pop_size, size=group_size)
        best_index = np.argmin([self.pop[i].fitness for i in indices])
        return self.pop[best_index]

    def save_pop_info(self, filename: str) -> None:
        """This method is meant to be used to
        save the final generation of individuals
        to a .csv file. But, it can be used at any
        point during evolution.
        The info that will be saved is lisp, fitness
        objective 1 (error), validation error, test
        error.

        Parameters
        ----------
        filename : str
            The location to save the file.
        """

        # Get data.
        fitness_all = [[ind.get_lisp_string(actual_lisp=True),
                        ind.fitness[0],
                        ind.validation_fitness,
                        ind.evaluate_test_points(self.test_dataset),
                        ind.fitness[1]] for ind in self.pop]

        # Save data.
        df = pd.DataFrame(fitness_all)

        df.to_csv(filename+'.csv', index=True,
                  header=['Equation (lisp)',
                          'Training Error',
                          'Validation Error',
                          'Testing Error'])

    def save(self, output_path: str,
             filename_ending: str) -> None:

        self.df_best_data.to_csv(os.path.join(output_path, 'best'+filename_ending),
                                 index=False)

        self.df_data_summary.to_csv(os.path.join(output_path, 'all_summary'+filename_ending),
                                    index=False)

    def update_best_individual(self) -> None:
        """Look through current population and if one
        individual has lower validation error than the current
        best (or there is not current best), update the best
        individual."""

        if self.best_individual is None:
            self.best_individual = self.pop[0]

        for p in self.pop:

            # Update best if an equation can beat it on valiation error
            if p.validation_fitness < self.best_individual.validation_fitness:
                self.best_individual = p

            # Update best if a shorted and equal accurate (on validation error)
            # equation is found.
            elif p.validation_fitness == self.best_individual.validation_fitness:
                if p.get_tree_size() < self.best_individual.get_tree_size():
                    self.best_individual = p

    def run(self,
            output_path: str = None):
        """Run the given number of generations with the given parameters.

        Parameters
        ----------
        output_path : str
            Location to save the data generated. This data includes
            summaries of fintesses and other measurements as well as
            individuals.
        """

        # initialize saving locations and names
        if output_path is None:
            output_path = os.path.join(os.environ['GP_DATA'], 'experiment'+str(self.exp))
        filename_ending = '_exp{}_rep{}_datasetindex{}.csv'.format(self.exp, self.rep, self.dataset_index)

        self.init_run(output_path)

        # for a fixed number of generations
        gen = 1
        while not self.stop(gen):

            # Do all the generation stuff --- mutate, evaluate...
            self.run_generation(gen=gen,
                                output_path=output_path,
                                num_to_mutate=self.pop_size)

            self.update_dataframes(gen=gen)
            print(gen, self.best_individual.fitness, self.best_individual.validation_fitness, self.best_individual.testing_fitness)
            print(self.best_individual.convert_lisp_to_standard())
            if gen % 1000 == 0:
                self.save(output_path, filename_ending)

            gen += 1

        self.save(output_path, filename_ending)

    def init_run(self, output_path):
        # initialize DataFrame for saving
        self.df_best_data = pd.DataFrame([],
                                         columns=['gen',
                                                  'train_error',
                                                  'val_error',
                                                  'test_error',
                                                  'lisp'])
        self.df_data_summary = pd.DataFrame([],
                                            columns=['gen',
                                                     'data_name',
                                                     'attribute',
                                                     'value'])

        # compute fitness for generation 0 (random individuals)
        self.compute_fitness()
        self.update_dataframes(gen=0)
        os.makedirs(output_path, exist_ok=True)

        print('gen, best train error, best validation error, best test error')
        print(0, self.best_individual.fitness, self.best_individual.validation_fitness, self.best_individual.testing_fitness)

    def stop(self, gen):
        if gen > self.max_gens:
            return True
        elif time.process_time() - self.start_time >= self.timeout:
            return True
        else:
            return False

    def update_dataframes(self, gen):

        # update summary for all individuals
        summary_info = self.get_summary_info()
        for data_name in summary_info:
            for attribute in summary_info[data_name]:
                row_data = [gen,
                            data_name,
                            attribute,
                            summary_info[data_name][attribute]]
                self.df_data_summary = self.append_to_df(df=self.df_data_summary,
                                                         row_data=row_data)

        # update best individual data
        self.update_best_individual()
        lisp = self.best_individual.get_lisp_string(actual_lisp=True)
        self.best_individual.evaluate_test_points(self.test_dataset)
        best_row_data = [gen, self.best_individual.fitness,
                         self.best_individual.validation_fitness,
                         self.best_individual.testing_fitness,
                         lisp]
        self.df_best_data = self.append_to_df(df=self.df_best_data,
                                              row_data=best_row_data)

    def append_to_df(self, df, row_data: List[Any]):
        new_row = pd.Series(row_data, index=df.columns)
        return df.append(new_row, ignore_index=True)
