import evoalgo.evolutionary_algorithms.genetic_algorithm as GA
import evoalgo.utils.genetic_operators as GenOp
from evoalgo.utils.evolution_process import Evolution
from evoalgo.utils.utils import Plotter
from evoalgo.optimization_problems.reinforcement_learning import max_gen_num, pop_size, params_num, task_name, \
    fitness_function, optimal_fit, is_torch  #, discrete_values_num  # import only when optimizing the string
import datetime

discrete_values_num = None

max_fit_history_dict = {}
avg_fit_history_dict = {}

selection_types = [('FPS', GenOp.Selection.fitness_proportionate),
                   ('STS', GenOp.Selection.stochastic_top_sampling),
                   ('Tour', GenOp.Selection.tournament)]
crossover_types = [('1PtCross', GenOp.Crossover.single_pt),
                   ('2PtCross', GenOp.Crossover.two_pt),
                   ('UniCross', GenOp.Crossover.uniform)]
mutation_types = [('DetMut', GenOp.Mutation.deterministic),
                  ('StoUniMut', GenOp.Mutation.stochastic_uniform),
                  ('GaussMut', GenOp.Mutation.gaussian_noise)]

algo_type = 'GA'


def run_algo(selection_f, crossover_f, mutation_f, key):
    if mutation_f.__name__ == GenOp.Mutation.gaussian_noise.__name__:
        ga = GA.SimpleGA(params_num, pop_size, discrete_values_num, is_torch=is_torch,
                         # mutation_var=0.5, sigma_min=0.01, sigma_decay=0.9)
                         mutation_var=0.5)
    else:
        ga = GA.SimpleGA(params_num, pop_size, discrete_values_num, is_torch=is_torch)

    Evolution.test_solver(ga, max_gen_num, task_name, fitness_function, selection_f, crossover_f, mutation_f,
                          plot_fit_history=False, print_fit_progress=False)
    max_fit_history_dict[key] = ga.pop_max_fit_history
    avg_fit_history_dict[key] = ga.pop_avg_fit_history


def compare():
    for selection_key, selection_f in selection_types:
        for crossover_key, crossover_f in crossover_types:
            for mutation_key, mutation_f in mutation_types:
                start_time = datetime.datetime.now()
                description = f'{selection_key} {crossover_key} {mutation_key}'
                run_algo(selection_f, crossover_f, mutation_f, description)
                print(f"{description} ~~~ Runtime: {str(datetime.datetime.now() - start_time).split('.')[0]}")

    Plotter.plot_fit_history_comparison(
        max_fit_history_dict, 'Max', max_gen_num, task_name, pop_size, optimal_fit, algo_type)
    Plotter.plot_fit_history_comparison(
        avg_fit_history_dict, 'Avg', max_gen_num, task_name, pop_size, optimal_fit, algo_type)


if __name__ == '__main__':
    compare()
