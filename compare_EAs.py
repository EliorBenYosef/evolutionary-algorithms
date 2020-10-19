import datetime

import evoalgo.evolutionary_algorithms.genetic_algorithm as GA
import evoalgo.evolutionary_algorithms.evolution_strategy as ES
import evoalgo.utils.genetic_operators as GenOp
from evoalgo.utils.evolution_process import Evolution
from evoalgo.utils.utils import colors_4, plot_fit_history_comparison
from evoalgo.optimization_problems.policy_nn import max_gen_num, pop_size, params_num, task_name, \
    fitness_function, optimal_fit  # , discrete_values_num  # import only when optimizing the string

discrete_values_num = None

max_fit_history_dict = {}
avg_fit_history_dict = {}


def compare_evolutionary_algorithms():
    """
    Compares ES algorithms (CMA-ES, OpenAI-ES, PEPG)
    and the best GA (Tournament selection, Uniform crossover, Gaussian mutation).
    """
    if discrete_values_num is not None:
        print('Evolution Strategy Algorithms are implemented only for real-number vector optimization')
        return

    start_time = datetime.datetime.now()
    cma_es = ES.CMA_ES(params_num, pop_size, sigma_init=0.5)
    Evolution.test_solver(cma_es, max_gen_num, task_name, fitness_function,
                          plot_fit_history=False, print_progress=True)
    print(f"CMA-ES ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")
    max_fit_history_dict['CMA-ES'] = cma_es.pop_max_fit_history
    avg_fit_history_dict['CMA-ES'] = cma_es.pop_avg_fit_history

    start_time = datetime.datetime.now()
    open_ai_es = ES.OpenAI_ES(params_num, pop_size, sigma_init=0.5, sigma_decay=1.0,  # sigma_decay=0.999
                              alpha_init=0.1, alpha_decay=1.0, antithetic_sampling=False, rank_fitness=False)
    Evolution.test_solver(open_ai_es, max_gen_num, task_name, fitness_function,
                          plot_fit_history=False, print_progress=True)
    print(f"OpenAI-ES ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")
    max_fit_history_dict['OpenAI-ES'] = open_ai_es.pop_max_fit_history
    avg_fit_history_dict['OpenAI-ES'] = open_ai_es.pop_avg_fit_history

    start_time = datetime.datetime.now()
    pepg = ES.PEPG(params_num, pop_size, sigma_init=0.5, sigma_decay=1.0,  # sigma_decay=0.999
                   alpha_init=0.1, alpha_decay=1.0, avg_fit_baseline=False, rank_fitness=False)
    Evolution.test_solver(pepg, max_gen_num, task_name, fitness_function,
                          plot_fit_history=False, print_progress=True)
    print(f"PEPG ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")
    max_fit_history_dict['PEPG'] = pepg.pop_max_fit_history
    avg_fit_history_dict['PEPG'] = pepg.pop_avg_fit_history

    start_time = datetime.datetime.now()
    ga = GA.SimpleGA(params_num, pop_size, mutation_var=1)  # sigma
    Evolution.test_solver(ga, max_gen_num, task_name, fitness_function,
                          GenOp.Selection.tournament, GenOp.Crossover.uniform, GenOp.Mutation.gaussian_noise,
                          plot_fit_history=False, print_progress=True)
    print(f"GA Tour UniCross GaussMut ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")
    max_fit_history_dict['GA Tour UniCross GaussMut'] = ga.pop_max_fit_history
    avg_fit_history_dict['GA Tour UniCross GaussMut'] = ga.pop_avg_fit_history

    plot_fit_history_comparison(
        max_fit_history_dict, 'Max', max_gen_num, task_name, pop_size, colors_4, optimal_fit)
    plot_fit_history_comparison(
        avg_fit_history_dict, 'Avg', max_gen_num, task_name, pop_size, colors_4, optimal_fit)


if __name__ == '__main__':
    compare_evolutionary_algorithms()
