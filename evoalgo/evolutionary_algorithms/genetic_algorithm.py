"""
Genetic Algorithms
"""

import numpy as np
from evoalgo.const import KEY_PARAMS_VEC, KEY_FITNESS
from evoalgo.utils.genetic_operators import Selection


class SimpleGA:
    """
    Simple Genetic Algorithm.
    """

    def __init__(self, params_num, pop_size, discrete_values_num=None, enable_elitism_selection=True,
                 selection_var=None, mutation_var=None, **kwargs):
        """
        :param params_num: number of model parameters
        :param pop_size: population size
        :param selection_var: top_size / tournament_size / truncation_size / elite_size
        :param mutation_var: mut_rate / sigma = sigma_init: initial STD
        :param kwargs: sigma_decay: anneal STD. sigma_decay=1 --> don't anneal the STD
        """
        self.params_num = params_num
        self.pop_size = pop_size
        self.enable_elitism_selection = enable_elitism_selection
        self.selection_var = selection_var
        self.mutation_var = mutation_var
        self.set_mutation_var(**kwargs)

        self.discrete_values_num = discrete_values_num

        self.population = None
        self.top_individual = None
        self.pop_avg_fit_history = []
        self.pop_max_fit_history = []

    def set_mutation_var(self, **kwargs):
        if kwargs.get('sigma_min'):
            self.sigma_min = kwargs.get('sigma_min')
        if kwargs.get('sigma_decay'):
            self.sigma_decay = kwargs.get('sigma_decay')

    def evolve(self, i, fitness_f, selection_f, crossover_f, mutation_f):
        self.init_pop() if i == 0 else self.update_pop(selection_f, crossover_f, mutation_f)  # TODO: remove: # solutions = solver.ask()
        self.eval_pop(fitness_f)

        if i != 0 and hasattr(self, 'sigma_decay') and hasattr(self, 'sigma_min'):
            if self.mutation_var > self.sigma_min:  # stop annealing if less than sigma_min
                self.mutation_var *= self.sigma_decay  # decay sigma.

    def init_pop(self):
        """
        initialize population (create the first generation).
        creates a population of agents (individuals),
        each is a dict that stores its NN parameters vector & fitness score
        """
        self.population = []  # solution models
        for i in range(self.pop_size):

            if self.discrete_values_num is not None:
                param = np.random.randint(low=0, high=self.discrete_values_num, size=self.params_num)
            else:
                # sample a random number from a standard normal distribution (mean 0, variance 1)
                param = np.random.randn(self.params_num)
                # the division gives a number which is close to 0 -> good for the NN weights.
                param /= 2.0  # TODO: test with 2.0, 10.0, and without

            self.population.append({KEY_PARAMS_VEC: param, KEY_FITNESS: None})

    def update_pop(self, selection_f, crossover_f, mutation_f):  # was called: next_gen
        """
        construct new population (create the next generation).
        :param selection_f: selection function
        :param crossover_f: crossover function
        :param mutation_f: mutation function
        """
        if self.enable_elitism_selection or \
                selection_f.__name__ == Selection.stochastic_universal_sampling.__name__ or \
                selection_f.__name__ == Selection.stochastic_top_sampling.__name__:
            # sort population by descending fitness-score:
            self.population.sort(key=lambda individual: individual[KEY_FITNESS], reverse=True)

        new_pop = []

        if self.enable_elitism_selection:
            new_pop.extend(Selection.elitism(self.population))

        while len(new_pop) < self.pop_size:
            p1_params, p2_params = \
                selection_f(self.population) if self.selection_var is None else \
                selection_f(self.population, self.selection_var)
            offspring = crossover_f(p1_params, p2_params, self.params_num)
            for o in offspring:
                mut_offspring = \
                    mutation_f(o, self.params_num, self.discrete_values_num) if self.mutation_var is None else \
                    mutation_f(o, self.params_num, self.discrete_values_num, self.mutation_var)
                new_pop.append({KEY_PARAMS_VEC: mut_offspring, KEY_FITNESS: None})

        self.population = new_pop

    def eval_pop(self, fitness_function):
        """
        evaluates all individuals and updates their fitness score
        :return: update population (with fitness scores), average population fitness
        """
        fit_sum = 0

        top_individual = None

        for individual in self.population:
            fit = fitness_function(individual[KEY_PARAMS_VEC])
            individual[KEY_FITNESS] = fit

            fit_sum += fit

            if top_individual is None or fit > top_individual[KEY_FITNESS]:
                top_individual = {
                    KEY_PARAMS_VEC: np.copy(individual[KEY_PARAMS_VEC]),
                    KEY_FITNESS: individual[KEY_FITNESS]
                }

        avg_fit = fit_sum / self.pop_size
        self.pop_avg_fit_history.append(avg_fit)

        self.pop_max_fit_history.append(top_individual[KEY_FITNESS])

        if self.top_individual is None or top_individual[KEY_FITNESS] > self.top_individual[KEY_FITNESS]:
            self.top_individual = {
                KEY_PARAMS_VEC: np.copy(top_individual[KEY_PARAMS_VEC]),
                KEY_FITNESS: top_individual[KEY_FITNESS]
            }
