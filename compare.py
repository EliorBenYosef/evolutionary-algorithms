import datetime

import evoalgo.evolutionary_algorithms.genetic_algorithm as GA
import evoalgo.evolutionary_algorithms.evolution_strategy as ES
import evoalgo.utils.genetic_operators as GenOp
from evoalgo.utils.evolution_process import Evolution
from evoalgo.utils.utils import Plotter
from evoalgo.optimization_problems.reinforcement_learning import max_gen_num, pop_size, params_num, task_name, \
    fitness_function, optimal_fit  # , discrete_values_num  # import only when optimizing the string

discrete_values_num = None

max_fit_history_dict = {}
avg_fit_history_dict = {}


def run_algo(selection_f, crossover_f, mutation_f, key):
    if mutation_f.__name__ == GenOp.Mutation.gaussian_noise.__name__:
        ga = GA.SimpleGA(params_num, pop_size, discrete_values_num,
                         # mutation_var=0.5, sigma_decay=0.999, sigma_min=0.01)
                         mutation_var=0.5)
    else:
        ga = GA.SimpleGA(params_num, pop_size, discrete_values_num)

    Evolution.test_solver(ga, max_gen_num, task_name, fitness_function,
                          selection_f, crossover_f, mutation_f,
                          plot_fit_history=False, print_fit_progress=False)
    max_fit_history_dict[key] = ga.pop_max_fit_history
    avg_fit_history_dict[key] = ga.pop_avg_fit_history


def compare_within_GA():
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

    for selection_key, selection_f in selection_types:
        for crossover_key, crossover_f in crossover_types:
            for mutation_key, mutation_f in mutation_types:
                start_time = datetime.datetime.now()
                description = f'{selection_key} {crossover_key} {mutation_key}'
                run_algo(selection_f, crossover_f, mutation_f, description)
                print(f"{description} ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")

    Plotter.plot_fit_history_comparison(
        max_fit_history_dict, 'Max', max_gen_num, task_name, pop_size, Plotter.colors_28, optimal_fit, algo_type)
    Plotter.plot_fit_history_comparison(
        avg_fit_history_dict, 'Avg', max_gen_num, task_name, pop_size, Plotter.colors_28, optimal_fit, algo_type)


def compare_ES_algorithms_and_best_GA():
    if discrete_values_num is not None:
        print('Evolution Strategy Algorithms are implemented only for real-number vector optimization')
        return

    start_time = datetime.datetime.now()
    cma_es = ES.CMA_ES(params_num, pop_size, sigma_init=0.5)
    Evolution.test_solver(cma_es, max_gen_num, task_name, fitness_function,
                          plot_fit_history=False, print_fit_progress=True)
    print(f"CMA-ES ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")
    max_fit_history_dict['CMA-ES'] = cma_es.pop_max_fit_history
    avg_fit_history_dict['CMA-ES'] = cma_es.pop_avg_fit_history

    start_time = datetime.datetime.now()
    open_ai_es = ES.OpenAI_ES(params_num, pop_size, sigma_init=0.5, sigma_decay=1.0,  # sigma_decay=0.999
                              alpha_init=0.1, alpha_decay=1.0, antithetic_sampling=False, rank_fitness=False)
    Evolution.test_solver(open_ai_es, max_gen_num, task_name, fitness_function,
                          plot_fit_history=False, print_fit_progress=True)
    print(f"OpenAI-ES ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")
    max_fit_history_dict['OpenAI-ES'] = open_ai_es.pop_max_fit_history
    avg_fit_history_dict['OpenAI-ES'] = open_ai_es.pop_avg_fit_history

    start_time = datetime.datetime.now()
    pepg = ES.PEPG(params_num, pop_size, sigma_init=0.5, sigma_decay=1.0,  # sigma_decay=0.999
                   alpha_init=0.1, alpha_decay=1.0, avg_fit_baseline=False, rank_fitness=False)
    Evolution.test_solver(pepg, max_gen_num, task_name, fitness_function,
                          plot_fit_history=False, print_fit_progress=True)
    print(f"PEPG / NES ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")
    max_fit_history_dict['PEPG / NES'] = pepg.pop_max_fit_history
    avg_fit_history_dict['PEPG / NES'] = pepg.pop_avg_fit_history

    start_time = datetime.datetime.now()
    ga = GA.SimpleGA(params_num, pop_size, mutation_var=1)  # sigma
    Evolution.test_solver(ga, max_gen_num, task_name, fitness_function,
                          GenOp.Selection.tournament, GenOp.Crossover.uniform, GenOp.Mutation.gaussian_noise,
                          plot_fit_history=False, print_fit_progress=True)
    print(f"GA Tour UniCross GaussMut ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")
    max_fit_history_dict['GA Tour UniCross GaussMut'] = ga.pop_max_fit_history
    avg_fit_history_dict['GA Tour UniCross GaussMut'] = ga.pop_avg_fit_history

    Plotter.plot_fit_history_comparison(
        max_fit_history_dict, 'Max', max_gen_num, task_name, pop_size, Plotter.colors_4, optimal_fit)
    Plotter.plot_fit_history_comparison(
        avg_fit_history_dict, 'Avg', max_gen_num, task_name, pop_size, Plotter.colors_4, optimal_fit)


if __name__ == '__main__':
    # compare_within_GA()
    compare_ES_algorithms_and_best_GA()
