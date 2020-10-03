from matplotlib import pyplot as plt
from os import path, mkdir
from const import KEY_PARAMS_VEC, KEY_FITNESS


class Evolution:

    @staticmethod
    def test_solver(solver, max_gen_num, task_name, fitness_f,
                    selection_f, crossover_f, mutation_f,
                    plot_fit_history=True):
        """
        uses solver to solve fit_func
        :param solver: an optimization method
        """
        for i in range(max_gen_num):
            solver.evolve(i, fitness_f, selection_f, crossover_f, mutation_f)
            if (i + 1) % 100 == 0 or i + 1 == max_gen_num:
                print("top fitness score at iteration:", (i + 1), solver.top_individual[KEY_FITNESS])

            # TODO: if implemented, change max_gen_num to i + 1 when plotting
            # if solver.top_individual[KEY_FITNESS] > REQUIRED_FITNESS_VALUE:  # stop the algorithm when target is reached.
            #     print(f'Required fitness ({str(REQUIRED_FITNESS_VALUE)}) reached at Gen {str(i + 1)}')
            #     break

        if plot_fit_history:
            Plotter.plot_fit_history(solver.pop_max_fit_history, 'Max', max_gen_num, task_name)
            Plotter.plot_fit_history(solver.pop_avg_fit_history, 'Average', max_gen_num, task_name)


class Plotter:
    colors = ['red', 'blue', 'orange', 'green']

    @staticmethod
    def plot_fit_history(fit_history, fit_history_type, max_gen_num, task_name):
        plt.title(task_name + ' - Population ' + fit_history_type + ' Fitness')
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        # x = [i + 1 for i in range(curr_gen + 1)]
        x = [i + 1 for i in range(max_gen_num)]
        plt.plot(x, fit_history)
        path_dir = 'results'
        if not path.isdir(path_dir):
            mkdir(path_dir)
        plt.savefig(path_dir + '/pop_' + fit_history_type + '_fit_' + task_name + '.png')
        plt.show()
        plt.close()

    @staticmethod
    def plot_fit_history_comparison(fit_history_dict, fit_history_type, max_gen_num, task_name, opt_fit=0):
        plt.figure(figsize=(16, 8), dpi=150)

        plt.title('Evolutionary Algorithms Comparison - Population ' + fit_history_type + ' Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')

        plt.xlim(0, max_gen_num)

        handles = []
        line_optimum, = plt.plot([opt_fit] * max_gen_num, color="black", linewidth=0.5, linestyle="-.", label='Global Optimum')
        handles.append(line_optimum)
        for i, (method_name, fit_history) in enumerate(fit_history_dict.items()):
            line, = plt.plot(fit_history, color=Plotter.colors[i], linewidth=1.0, linestyle="-", label=method_name)
            handles.append(line)

        plt.legend(handles=handles, loc=4)

        path_dir = 'results'
        if not path.isdir(path_dir):
            mkdir(path_dir)
        plt.savefig(path_dir + '/pop_' + fit_history_type + '_fit_' + task_name + '.png')

        # plt.show()
        plt.close()


class Printer:

    @staticmethod
    def print_top_individual(pop):
        pop.sort(key=lambda individual: individual[KEY_FITNESS], reverse=True)  # descending fitness score
        print(f"Top individual's fitness: {str(pop[0][KEY_FITNESS])}")
        print(f"Top individual's params: {str(pop[0][KEY_PARAMS_VEC])}")
