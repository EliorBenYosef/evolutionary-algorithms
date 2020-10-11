from evoalgo.const import KEY_FITNESS
from evoalgo.utils.utils import Plotter


class Evolution:

    @staticmethod
    def test_solver(solver, max_gen_num, task_name, fitness_f,
                    selection_f=None, crossover_f=None, mutation_f=None,
                    plot_fit_history=True, print_fit_progress=True):
        """
        uses solver (evolutionary algorithm) to solve fit_func.
        :param solver: an optimization method
        """
        for i in range(max_gen_num):

            if selection_f is None and crossover_f is None and mutation_f is None:
                solver.evolve(fitness_f)
            else:
                solver.evolve(i, fitness_f, selection_f, crossover_f, mutation_f)

            if print_fit_progress and ((i + 1) % 100 == 0 or i + 1 == max_gen_num):
                print(f"top fitness score at iteration {(i + 1)} : {solver.top_individual[KEY_FITNESS]}")

            # TODO: if implemented, change max_gen_num to i + 1 when plotting
            # if solver.top_individual[KEY_FITNESS] > REQUIRED_FITNESS_VALUE:  # stop the algorithm when target is reached.
            #     print(f"Required fitness ({str(REQUIRED_FITNESS_VALUE)}) reached at Gen {str(i + 1)}")
            #     break

        if plot_fit_history:
            Plotter.plot_fit_history(solver.pop_max_fit_history, 'Max', max_gen_num, task_name)
            Plotter.plot_fit_history(solver.pop_avg_fit_history, 'Average', max_gen_num, task_name)
