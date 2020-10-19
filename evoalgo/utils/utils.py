from os import path, mkdir
from matplotlib import pyplot as plt

from evoalgo.const import KEY_PARAMS_VEC, KEY_FITNESS


# Plotter:

colors_4 = ['red', 'orange', 'green', 'blue']

colors_28 = ['#f00082', '#f000bf', '#ff75e3',  # pink
             '#FF0000', '#fa3c3c', '#fa7a7a', '#ff2b0a',  # red
             '#FF7F00', '#ff6400', '#ffa538',  # orange
             '#FFFF00', '#e6dc32', '#fff76b',  # yellow
             '#00FF00', '#a0e632', '#00dc00', '#17A858', '#00d28c',  # green
             '#0000FF', '#00c8c8', '#0DB0DD', '#00a0ff', '#1e3cff',  # blue
             '#4B0082', '#a000c8', '#6e00dc', '#8B00FF', '#9400D3']  # purple


def plot_fit_history(fit_history, fit_history_type, max_gen_num, task_name):
    plt.title(f"{task_name} - Population {fit_history_type} Fitness")
    plt.ylabel(f"{fit_history_type} Fitness")
    plt.xlabel('Generation')

    plt.xlim(0, max_gen_num)

    # x = [i + 1 for i in range(curr_gen + 1)]
    x = [i + 1 for i in range(max_gen_num)]
    plt.plot(x, fit_history)

    path_dir = 'results'
    if not path.isdir(path_dir):
        mkdir(path_dir)
    plt.savefig(f'{path_dir}/pop_{fit_history_type}_fit_{task_name}.png')

    plt.show()
    plt.close()


def plot_fit_history_comparison(fit_history_dict, fit_history_type, max_gen_num, task_name, pop_size, colors,
                                optimal_fit=None, algo_type='Evolutionary Algorithms'):
    plt.figure(figsize=(16, 8), dpi=150)

    plt.title(f"{task_name} - {algo_type} Comparison - Population: {str(pop_size)} - {fit_history_type} Fitness")
    plt.ylabel(f"{fit_history_type} Fitness")
    plt.xlabel('Generation')

    plt.xlim(0, max_gen_num)

    handles = []
    if optimal_fit is not None:
        line_optimum, = plt.plot([optimal_fit] * max_gen_num, color="black", linewidth=0.5, linestyle="-.", label='Global Optimum')
        handles.append(line_optimum)
    for i, (method_name, fit_history) in enumerate(fit_history_dict.items()):
        line, = plt.plot(fit_history, color=colors[i], linewidth=1.0, linestyle="-", label=method_name)
        # line, = plt.plot(fit_history, linewidth=1.0, linestyle="-", label=method_name)
        handles.append(line)

    plt.legend(handles=handles, bbox_to_anchor=(1.01, 0), loc="lower left")

    path_dir = 'results'
    if not path.isdir(path_dir):
        mkdir(path_dir)
    plt.savefig(f'{path_dir}/{task_name}-Pop{str(pop_size)}-{fit_history_type}.png', bbox_inches='tight')

    # plt.show()
    plt.close()


# Printer:

def print_top_individual(pop):
    pop.sort(key=lambda individual: individual[KEY_FITNESS], reverse=True)  # descending fitness score
    print(f"Top individual's fitness: {str(pop[0][KEY_FITNESS])}")
    print(f"Top individual's params: {str(pop[0][KEY_PARAMS_VEC])}")
