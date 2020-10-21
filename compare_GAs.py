import datetime

import evoalgo.evolutionary_algorithms.genetic_algorithm as GA
import evoalgo.utils.genetic_operators as GenOp
from evoalgo.utils.evolution_process import Evolution
from evoalgo.utils.utils import colors_28, plot_fit_history_comparison
from evoalgo.optimization_problems.policy_nn import max_gen_num, pop_size, params_num, task_name, \
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
                          print_progress=False, plot=False)
    max_fit_history_dict[key] = ga.pop_max_fit_history
    avg_fit_history_dict[key] = ga.pop_avg_fit_history


def compare_genetic_algorithms():
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

    plot_fit_history_comparison(max_fit_history_dict, 'Max', max_gen_num, task_name, pop_size, optimal_fit, algo_type,
                                colors=colors_28)
    plot_fit_history_comparison(avg_fit_history_dict, 'Avg', max_gen_num, task_name, pop_size, optimal_fit, algo_type,
                                colors=colors_28)


if __name__ == '__main__':
    compare_genetic_algorithms()
