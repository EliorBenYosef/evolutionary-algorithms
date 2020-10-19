import evoalgo.evolutionary_algorithms.genetic_algorithm as GA
import evoalgo.utils.genetic_operators as GenOp
from evoalgo.utils.evolution_process import Evolution
from evoalgo.utils.utils import Plotter

from evoalgo.optimization_problems.string_opt import params_num as params_num_str, task_name as task_name_str, \
    fitness_function as fitness_function_str, optimal_fit as optimal_fit_str, \
    discrete_values_num as discrete_values_num_str
from evoalgo.optimization_problems.rastrigin import params_num as params_num_rst, task_name as task_name_rst, \
    fitness_function as fitness_function_rst, optimal_fit as optimal_fit_rst
from evoalgo.optimization_problems.policy_nn import params_num as params_num_nn, task_name as task_name_nn, \
    fitness_function as fitness_function_nn, optimal_fit as optimal_fit_nn


def run_genetic_algorithms(max_gen_num, pop_size, params_num, task_name,
                           fitness_function, optimal_fit, discrete_values_num=None):
    algo_type = 'GA'

    selection_types = [('FPS', GenOp.Selection.fitness_proportionate),
                       ('STS', GenOp.Selection.stochastic_top_sampling),
                       ('Tour', GenOp.Selection.tournament)]
    crossover_types = [('1PtCross', GenOp.Crossover.single_pt),
                       ('2PtCross', GenOp.Crossover.two_pt),
                       ('UniCross', GenOp.Crossover.uniform)]
    mutation_types = [('DetMut', GenOp.Mutation.deterministic),
                      ('StoUniMut', GenOp.Mutation.stochastic_uniform),
                      ('GaussMut', GenOp.Mutation.gaussian_noise)]

    max_fit_history_dict = {}
    avg_fit_history_dict = {}

    for selection_key, selection_f in selection_types:
        for crossover_key, crossover_f in crossover_types:
            for mutation_key, mutation_f in mutation_types:

                if mutation_f.__name__ == GenOp.Mutation.gaussian_noise.__name__:
                    ga = GA.SimpleGA(params_num, pop_size, discrete_values_num,
                                     # mutation_var=0.5, sigma_decay=0.999, sigma_min=0.01)
                                     mutation_var=0.5)
                else:
                    ga = GA.SimpleGA(params_num, pop_size, discrete_values_num)

                Evolution.test_solver(ga, max_gen_num, task_name, fitness_function,
                                      selection_f, crossover_f, mutation_f,
                                      plot_fit_history=True, print_fit_progress=True)

                description = f'{selection_key} {crossover_key} {mutation_key}'
                max_fit_history_dict[description] = ga.pop_max_fit_history
                avg_fit_history_dict[description] = ga.pop_avg_fit_history

    Plotter.plot_fit_history_comparison(
        max_fit_history_dict, 'Max', max_gen_num, task_name, pop_size, Plotter.colors_28, optimal_fit, algo_type)
    Plotter.plot_fit_history_comparison(
        avg_fit_history_dict, 'Avg', max_gen_num, task_name, pop_size, Plotter.colors_28, optimal_fit, algo_type)


def test_optimization_problems():
    """
    Runs the algorithms with a population of 100, for 100 generation.
    """
    run_genetic_algorithms(100, 100, params_num_str, task_name_str, fitness_function_str, optimal_fit_str,
                           discrete_values_num_str)
    run_genetic_algorithms(100, 100, params_num_rst, task_name_rst, fitness_function_rst, optimal_fit_rst)
    run_genetic_algorithms(100, 100, params_num_nn, task_name_nn, fitness_function_nn, optimal_fit_nn)
