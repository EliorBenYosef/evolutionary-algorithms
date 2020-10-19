import evoalgo.evolutionary_algorithms.evolution_strategy as ES
from evoalgo.utils.evolution_process import Evolution
from evoalgo.utils.utils import colors_4, plot_fit_history_comparison

from evoalgo.optimization_problems.rastrigin import params_num as params_num_rst, task_name as task_name_rst, \
    fitness_function as fitness_function_rst, optimal_fit as optimal_fit_rst
from evoalgo.optimization_problems.policy_nn import params_num as params_num_nn, task_name as task_name_nn, \
    fitness_function as fitness_function_nn, optimal_fit as optimal_fit_nn


def run_oes(max_gen_num, pop_size, params_num, task_name, fitness_function,
            antithetic_sampling=False, rank_fitness=False):
    open_ai_es = ES.OpenAI_ES(params_num, pop_size, sigma_init=0.5, sigma_decay=1.0,  # sigma_decay=0.999
                              alpha_init=0.1, alpha_decay=1.0,
                              antithetic_sampling=antithetic_sampling, rank_fitness=rank_fitness)
    Evolution.test_solver(open_ai_es, max_gen_num, task_name, fitness_function,
                          plot_fit_history=True, print_progress=True)
    return open_ai_es


def run_pepg(max_gen_num, pop_size, params_num, task_name, fitness_function,
             avg_fit_baseline=False, rank_fitness=False):
    pepg = ES.PEPG(params_num, pop_size, sigma_init=0.5, sigma_decay=1.0,  # sigma_decay=0.999
                   alpha_init=0.1, alpha_decay=1.0,
                   avg_fit_baseline=avg_fit_baseline, rank_fitness=rank_fitness)
    Evolution.test_solver(pepg, max_gen_num, task_name, fitness_function,
                          plot_fit_history=True, print_progress=True)
    return pepg


def run_evolution_strategy_algorithms(max_gen_num, pop_size, params_num, task_name,
                                      fitness_function, optimal_fit, discrete_values_num=None):

    if discrete_values_num is not None:
        print('Evolution Strategy Algorithms are implemented only for real-number vector optimization')
        return

    max_fit_history_dict = {}
    avg_fit_history_dict = {}

    ##############################

    # CMA-ES
    cma_es = ES.CMA_ES(params_num, pop_size, sigma_init=0.5)
    Evolution.test_solver(cma_es, max_gen_num, task_name, fitness_function,
                          plot_fit_history=True, print_progress=True)
    max_fit_history_dict['CMA-ES'] = cma_es.pop_max_fit_history
    avg_fit_history_dict['CMA-ES'] = cma_es.pop_avg_fit_history

    ##############################

    # OpenAI-ES
    open_ai_es = run_oes(max_gen_num, pop_size, params_num, task_name, fitness_function,
                         antithetic_sampling=False, rank_fitness=False)
    max_fit_history_dict['OpenAI-ES'] = open_ai_es.pop_max_fit_history
    avg_fit_history_dict['OpenAI-ES'] = open_ai_es.pop_avg_fit_history

    open_ai_es_r = run_oes(max_gen_num, pop_size, params_num, task_name, fitness_function,
                           antithetic_sampling=False, rank_fitness=True)
    max_fit_history_dict['OpenAI-ES rank'] = open_ai_es_r.pop_max_fit_history
    avg_fit_history_dict['OpenAI-ES rank'] = open_ai_es_r.pop_avg_fit_history

    open_ai_es_a = run_oes(max_gen_num, pop_size, params_num, task_name, fitness_function,
                           antithetic_sampling=True, rank_fitness=False)
    max_fit_history_dict['OpenAI-ES antithetic'] = open_ai_es_a.pop_max_fit_history
    avg_fit_history_dict['OpenAI-ES antithetic'] = open_ai_es_a.pop_avg_fit_history

    open_ai_es_r_a = run_oes(max_gen_num, pop_size, params_num, task_name, fitness_function,
                             antithetic_sampling=True, rank_fitness=True)
    max_fit_history_dict['OpenAI-ES rank antithetic'] = open_ai_es_r_a.pop_max_fit_history
    avg_fit_history_dict['OpenAI-ES rank antithetic'] = open_ai_es_r_a.pop_avg_fit_history

    ##############################

    # PEPG
    pepg = run_pepg(max_gen_num, pop_size, params_num, task_name, fitness_function,
                    avg_fit_baseline=False, rank_fitness=False)
    max_fit_history_dict['PEPG'] = pepg.pop_max_fit_history
    avg_fit_history_dict['PEPG'] = pepg.pop_avg_fit_history

    pepg_r = run_pepg(max_gen_num, pop_size, params_num, task_name, fitness_function,
                      avg_fit_baseline=False, rank_fitness=True)
    max_fit_history_dict['PEPG rank'] = pepg_r.pop_max_fit_history
    avg_fit_history_dict['PEPG rank'] = pepg_r.pop_avg_fit_history

    pepg_a = run_pepg(max_gen_num, pop_size, params_num, task_name, fitness_function,
                      avg_fit_baseline=True, rank_fitness=False)
    max_fit_history_dict['PEPG avg_base'] = pepg_a.pop_max_fit_history
    avg_fit_history_dict['PEPG avg_base'] = pepg_a.pop_avg_fit_history

    pepg_r_a = run_pepg(max_gen_num, pop_size, params_num, task_name, fitness_function,
                        avg_fit_baseline=True, rank_fitness=True)
    max_fit_history_dict['PEPG rank avg_base'] = pepg_r_a.pop_max_fit_history
    avg_fit_history_dict['PEPG rank avg_base'] = pepg_r_a.pop_avg_fit_history

    ##############################

    plot_fit_history_comparison(
        max_fit_history_dict, 'Max', max_gen_num, task_name, pop_size, colors_4, optimal_fit)
    plot_fit_history_comparison(
        avg_fit_history_dict, 'Avg', max_gen_num, task_name, pop_size, colors_4, optimal_fit)


def test_optimization_problems():
    """
    Runs the algorithms with a population of 100, for 100 generation.
    """
    run_evolution_strategy_algorithms(100, 100, params_num_rst, task_name_rst, fitness_function_rst, optimal_fit_rst)
    run_evolution_strategy_algorithms(100, 100, params_num_nn, task_name_nn, fitness_function_nn, optimal_fit_nn)
