"""
Optimizing: Rastrigin function params.
"""

import numpy as np

max_gen_num = 1000  # max_iter_num # number of generations to run each solver  # TODO: 5000
pop_size = 100

params_num = 100  # number of model parameters (expresses the problem's dimensionality)

task_name = 'Rastrigin' + str(params_num) + 'D'
# optimal_fit = 0  # global maximum point  # TODO: uncomment when optimization problem is solved
optimal_fit = None  # TODO: remove when optimization problem is solved


def rastrigin_function(x):
    """
    taken from: https://github.com/CMA-ES/pycma/blob/master/cma/fitness_functions.py
    """
    if not np.isscalar(x[0]):
        N = len(x[0])
        return np.array([10 * N + sum(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
    N = len(x)
    return 10 * N + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def fitness_function(individual_params):
    """
    modifies the Rastrigin function:
    * -10. units shift - to move the optimum point away from origin.
    * sign flip - to have a global maximum (instead of a global minimum).
    :return: individual's fitness score
    """
    individual_params = np.copy(individual_params)
    individual_params -= 10.0  # -10. units shift
    return -rastrigin_function(individual_params)  # sign flip


def test_fitness_function():
    x = np.zeros(params_num)
    print(f"F(zeros(params_num)) = {fitness_function(x)}")
    x = np.ones(params_num) * 10.
    print(f"F(ones(params_num) * 10) = {fitness_function(x)}")


if __name__ == '__main__':
    test_fitness_function()
