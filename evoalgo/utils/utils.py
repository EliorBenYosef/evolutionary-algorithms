import os
from matplotlib import pyplot as plt

from evoalgo.const import KEY_PARAMS_VEC, KEY_FITNESS


# Plotter:

colors_28 = ['#f00082', '#f000bf', '#ff75e3',  # pink
             '#FF0000', '#fa3c3c', '#fa7a7a', '#ff2b0a',  # red
             '#FF7F00', '#ff6400', '#ffa538',  # orange
             '#FFFF00', '#e6dc32', '#fff76b',  # yellow
             '#00FF00', '#a0e632', '#00dc00', '#17A858', '#00d28c',  # green
             '#0000FF', '#00c8c8', '#0DB0DD', '#00a0ff', '#1e3cff',  # blue
             '#4B0082', '#a000c8', '#6e00dc', '#8B00FF', '#9400D3']  # purple


def plot_fit_history(fit_history, fit_history_type, max_gen_num, task_name, show=False, save=True):
    plt.title(f"{task_name} - Population {fit_history_type} Fitness")
    plt.ylabel(f"{fit_history_type} Fitness")
    plt.xlabel('Generation')

    plt.xlim(0, max_gen_num)

    # x = [i + 1 for i in range(curr_gen + 1)]
    x = [i + 1 for i in range(max_gen_num)]
    plt.plot(x, fit_history)

    if save:
        path_dir = 'results'
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)
        plt.savefig(f'{path_dir}/pop_{fit_history_type}_fit_{task_name}.png')

    if show:
        plt.show()

    plt.close()


def plot_fit_history_comparison(fit_history_dict, fit_history_type, max_gen_num, task_name, pop_size,
                                optimal_fit=None, algo_type='Evolutionary Algorithms',
                                colors=None, show=False, save=True):

    plt.figure(figsize=(16, 8), dpi=150)

    plt.title(f"{task_name} - {algo_type} Comparison - Population: {str(pop_size)} - {fit_history_type} Fitness")
    plt.ylabel(f"{fit_history_type} Fitness")
    plt.xlabel('Generation')

    plt.xlim(0, max_gen_num)

    x = [i + 1 for i in range(max_gen_num)]

    handles = []
    if optimal_fit is not None:
        line_optimum, = plt.plot(x, [optimal_fit] * max_gen_num, label='Global Optimum',
                                 linewidth=0.5, linestyle="-.", color="black")
        handles.append(line_optimum)
    for i, (method_name, fit_history) in enumerate(fit_history_dict.items()):
        if colors is not None:
            line, = plt.plot(x, fit_history, label=method_name, linewidth=1.0, linestyle="-", color=colors[i])
        else:
            line, = plt.plot(x, fit_history, label=method_name, linewidth=1.0, linestyle="-")  # auto colors

        handles.append(line)

    plt.legend(handles=handles, bbox_to_anchor=(1.01, 0), loc="lower left")

    if save:
        path_dir = 'results'
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)
        plt.savefig(f'{path_dir}/{task_name}-Pop{str(pop_size)}-{fit_history_type}.png', bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


# Printer:

def print_top_individual(pop):
    pop.sort(key=lambda individual: individual[KEY_FITNESS], reverse=True)  # descending fitness score
    print(f"Top individual's fitness: {str(pop[0][KEY_FITNESS])}")
    print(f"Top individual's params: {str(pop[0][KEY_PARAMS_VEC])}")
