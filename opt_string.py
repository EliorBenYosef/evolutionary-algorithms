"""
Optimizing: String characters (discrete-valued vector).
the String is converted into a discrete-number vector.

Evolving a population of (fixed-length) random strings toward a chosen (same-length) target string.

Running the algorithm:
* Using a 12 characters target string - should take a few minutes on a modern CPU.
* Using a longer (>12 characters) target string - will take much more time and resources to evolve.
"""

import random
from difflib import SequenceMatcher
from matplotlib import pyplot as plt
from os import path, mkdir

from algo_GA import SimpleGA
from algo_ES import CMA_ES, PEPG, OpenAI_ES
from utils import Evolution, Plotter
from util_GeneticOperators import Selection, Crossover, Mutation

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.! "
discrete_values_num = 56  # 26+26+4=56 characters

target = "FullyEvolved"  # 12-character string
params_num = len(target)  # target string length
max_gen_num = 150
pop_size = 900

task_name = 'String'
opt_fit = 1


#########################################

def convert_indices_to_string(individual_params):
    individual_string = ''
    for i in individual_params:
        individual_string += alphabet[i]
    return individual_string


def fitness_function(individual_params):
    """
    computes the strings' similarity
    :param individual_params: discrete-numbers vector
    :return: individual's fitness score
    """
    individual_string = convert_indices_to_string(individual_params)
    return SequenceMatcher(None, individual_string, target).ratio()


#########################################

max_fit_history_dict = {}
avg_fit_history_dict = {}

ga_fps = SimpleGA(params_num, pop_size, discrete_values_num, mutation_var=1e-2)
Evolution.test_solver(ga_fps, max_gen_num, task_name, fitness_function, plot_fit_history=False,
                      selection_f=Selection.fitness_proportionate,
                      crossover_f=Crossover.uniform,
                      mutation_f=Mutation.stochastic_uniform)
max_fit_history_dict['GA_fps'] = ga_fps.pop_max_fit_history
avg_fit_history_dict['GA_fps'] = ga_fps.pop_avg_fit_history

ga_sts = SimpleGA(params_num, pop_size, discrete_values_num, selection_var=0.1, mutation_var=1e-2)
Evolution.test_solver(ga_sts, max_gen_num, task_name, fitness_function, plot_fit_history=False,
                      selection_f=Selection.stochastic_top_sampling,
                      crossover_f=Crossover.uniform,
                      mutation_f=Mutation.stochastic_uniform)
max_fit_history_dict['GA_sts'] = ga_sts.pop_max_fit_history
avg_fit_history_dict['GA_sts'] = ga_sts.pop_avg_fit_history

ga_tournament = SimpleGA(params_num, pop_size, discrete_values_num, selection_var=0.2, mutation_var=1e-2)
Evolution.test_solver(ga_tournament, max_gen_num, task_name, fitness_function, plot_fit_history=False,
                      selection_f=Selection.tournament,
                      crossover_f=Crossover.single_pt,
                      mutation_f=Mutation.stochastic_uniform)
max_fit_history_dict['GA_tournament'] = ga_tournament.pop_max_fit_history
avg_fit_history_dict['GA_tournament'] = ga_tournament.pop_avg_fit_history

Plotter.plot_fit_history_comparison(max_fit_history_dict, 'Max', max_gen_num, task_name, opt_fit)
Plotter.plot_fit_history_comparison(avg_fit_history_dict, 'Avg', max_gen_num, task_name, opt_fit)

