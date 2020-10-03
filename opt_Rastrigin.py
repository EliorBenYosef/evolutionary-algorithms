"""
Optimizing: Rastrigin function params.
"""

import numpy as np

from algo_GA import SimpleGA
from algo_ES import CMA_ES, PEPG, OpenAI_ES
from utils import Evolution, Plotter
from util_GeneticOperators import Selection, Crossover, Mutation


max_gen_num = 1000  # max_iter_num # number of generations to run each solver # TODO: 5000
pop_size = 100

params_num = 100  # number of model parameters (expresses the problem's dimensionality)


########################################

def rastrigin_function(x):
    """
    taken from: https://github.com/CMA-ES/pycma/blob/master/cma/fitness_functions.py
    """
    if not np.isscalar(x[0]):
        N = len(x[0])
        return np.array([10 * N + sum(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
    N = len(x)
    return 10 * N + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def fitness_function(individual_params):
    """
    modifies the Rastrigin function:
    * -10. units shift - to move the optimum point away from origin.
    * sign flip - to have a global maximum (instead of a global minimum).
    :return: individual's fitness score
    """
    individual_params = np.copy(individual_params)
    individual_params -= 10.0  # -10. units shift
    return -rastrigin_function(individual_params)  # sign flip


# # testing the fitness function:
# x = np.zeros(params_num)
# print("F(x) = ", fit_func(x))
# x = np.ones(params_num) * 10.
# print("F(x) = ", fit_func(x))


########################################

# fit_history_dict = {}
#
# cma_es = CMA_ES(params_num, pop_size,
#                 weight_decay=0.0,
#                 sigma_init=0.5)
# history_cma_es = test_solver(cma_es)
# fit_history_dict['CMA-ES'] = cma_es.pop_max_fit_history
#
# pepg = PEPG(params_num, pop_size,
#             sigma_init=0.5,
#             alpha=0.1, alpha_decay=1.0,  # don't anneal the learning rate
#             weight_decay=0.00,
#             average_baseline=False, rank_fitness=False, forget_best=False)
# history_pepg = test_solver(pepg)
# fit_history_dict['PEPG / NES'] = pepg.pop_max_fit_history
#
# open_ai_es = OpenAI_ES(params_num, pop_size,
#                        sigma_init=0.5, sigma_decay=0.999,  # don't anneal the STD
#                        alpha=0.1, alpha_decay=1.0,
#                        weight_decay=0.00,
#                        antithetic=False, rank_fitness=False, forget_best=False)
# history_open_ai_es = test_solver(open_ai_es)
# fit_history_dict['OpenAI-ES'] = open_ai_es.pop_max_fit_history

ga = SimpleGA(params_num, pop_size, selection_var=0.1, mutation_var=0.5,
              sigma_min=0.01, sigma_decay=0.999)  # sigma=0.1
Evolution.test_solver(ga, max_gen_num, 'Rastrigin_' + str(params_num) + 'D', fitness_function,
                      selection_f=Selection.stochastic_top_sampling,
                      crossover_f=Crossover.uniform,
                      mutation_f=Mutation.gaussian_noise)
# fit_history_dict['GA'] = ga.pop_max_fit_history
#
# Plotter.plot_fit_history_comparison(fit_history_dict, 'Max', max_gen_num, file_name='rastrigin_' + str(params_num) + 'd')
