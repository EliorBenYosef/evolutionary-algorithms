"""
Evolution Strategy Algorithms.
Implemented only for real-number vector optimization.
"""

from cma import CMAEvolutionStrategy

import numpy as np
from evoalgo.const import KEY_PARAMS_VEC, KEY_FITNESS
import evoalgo.utils.optimizers as optimizers
from evoalgo.utils.fitness_shaping import compute_zero_centered_ranks
import evoalgo.utils.genetic_operators as GenOp


class CMA_ES:
    """
    CMA-ES algorithm (cma's CMAEvolutionStrategy) wrapper.
    """

    def __init__(self, params_num, pop_size, sigma_init=0.1):
        """
        :param params_num: number of model parameters
        :param pop_size: population size
        :param sigma_init: mut_rate / sigma = sigma_init: initial STD
        """
        self.params_num = params_num
        self.pop_size = pop_size

        self.sigma_init = sigma_init
        self.sigma = sigma_init

        self.cma_es = CMAEvolutionStrategy(self.params_num * [0], self.sigma_init, {'popsize': self.pop_size})

        self.population = None
        self.fitness_scores = None
        self.top_individual = None
        self.pop_avg_fit_history = []
        self.pop_max_fit_history = []

    def evolve(self, fitness_f):
        self.population = np.array(self.cma_es.ask())
        self.eval_pop(fitness_f)

    def eval_pop(self, fitness_function):
        self.fitness_scores = np.zeros(self.pop_size)
        for i, params in enumerate(self.population):
            self.fitness_scores[i] = fitness_function(params)

        avg_fit = np.sum(self.fitness_scores) / self.pop_size
        self.pop_avg_fit_history.append(avg_fit)

        self.cma_es.tell(self.population, (-self.fitness_scores).tolist())  # sign flip since the optimizer is a minimizer

        result = self.cma_es.result
        top_individual_params = result[0]  # best_mu
        top_individual_fitness = -result[1]
        # top_individual_index = result[2] - 1
        self.sigma = result[6]

        self.pop_max_fit_history.append(top_individual_fitness)

        self.top_individual = {
            KEY_PARAMS_VEC: top_individual_params,
            KEY_FITNESS: top_individual_fitness
        }


class OpenAI_ES:
    """
    OpenAI's ES algorithm (basic version).
    """

    def __init__(self, params_num, pop_size,
                 sigma_init=0.1, sigma_decay=0.999, sigma_min=0.01,
                 alpha_init=0.01, alpha_decay=0.9999, alpha_min=0.001,
                 antithetic_sampling=False, rank_fitness=True):
        """
        :param params_num: number of model parameters
        :param pop_size: population size
        :param sigma_init: initial STD
        :param sigma_decay: anneal STD. sigma_decay=1 --> don't anneal the STD
        :param sigma_min: stop annealing if less than this
        :param alpha_init: learning rate for STD
        :param alpha_decay: annealing the learning rate. alpha_decay=1.0 --> don't anneal the learning rate
        :param alpha_min: stop annealing learning rate
        :param antithetic_sampling: antithetic sampling of epsilon (gaussian-noise vector).
               half of the population have some params, and the other half has the opposite params (sign change)
        :param rank_fitness: rank-based fitness-shaping - use rank rather than fitness numbers.
        """
        self.params_num = params_num
        self.pop_size = pop_size
        self.rank_fitness = rank_fitness

        ####################################

        # new population's params variables:
        self.mu = np.zeros(self.params_num)

        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min

        self.epsilon = None
        self.antithetic_sampling = antithetic_sampling
        if self.antithetic_sampling:
            if self.pop_size % 2 != 0:  # Antithetic sampling requires even population size
                self.pop_size += 1

        ####################################

        self.alpha = alpha_init
        self.alpha_init = alpha_init
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.optimizer = optimizers.Adam(pi=self, step_size=alpha_init)  # updates self.mu

        ####################################

        self.population = None
        self.fitness_scores = None
        self.top_individual = None
        self.pop_avg_fit_history = []
        self.pop_max_fit_history = []

    def evolve(self, fitness_f):
        self.update_pop()
        self.eval_pop(fitness_f)
        self.decay_variables()

    def update_pop(self):
        self.epsilon = GenOp.Mutation.sample_epsilon(self.pop_size, self.params_num, self.antithetic_sampling)
        self.population = self.mu[None, :] + self.epsilon * self.sigma

    def eval_pop(self, fitness_function):
        self.fitness_scores = np.zeros(self.pop_size)
        for i, params in enumerate(self.population):
            self.fitness_scores[i] = fitness_function(params)

        avg_fit = np.sum(self.fitness_scores) / self.pop_size
        self.pop_avg_fit_history.append(avg_fit)

        fit_arr = compute_zero_centered_ranks(self.fitness_scores) if self.rank_fitness else self.fitness_scores

        indices_sort = np.argsort(fit_arr)[::-1]  # sorted by fitness score \ fitness rank
        top_individual_params = self.population[indices_sort[0]]
        top_individual_fitness = self.fitness_scores[indices_sort[0]]
        self.pop_max_fit_history.append(top_individual_fitness)

        if self.top_individual is None or top_individual_fitness > self.top_individual[KEY_FITNESS]:
            self.top_individual = {
                KEY_PARAMS_VEC: top_individual_params,
                KEY_FITNESS: top_individual_fitness
            }

        self.update_mu(fit_arr)

    def update_mu(self, fit_arr):
        standardized_fit = (fit_arr - np.mean(fit_arr)) / np.std(fit_arr)
        # formula: ∆μ = α ∙ 1/(Nσ) ∙ ∑(Fi∙ei),  i=1,...,N
        delta_mu = 1. / (self.pop_size * self.sigma) * np.dot(standardized_fit, self.epsilon)  # Population averaging

        if self.optimizer is not None:
            # the optimizer updates self.mu within itself
            self.optimizer.step_size = self.alpha
            update_ratio = self.optimizer.update(-delta_mu)  # sign flip since the optimizer is a minimizer
        else:
            self.mu += self.alpha * delta_mu  # normal SGD method?

    def decay_variables(self):
        if self.sigma_decay < 1 and self.sigma > self.sigma_min:
            self.sigma *= self.sigma_decay

        if self.alpha_decay < 1 and self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay


class PEPG:
    """
    PEPG (NES) (extension).
    """

    def __init__(self, params_num, pop_size,
                 sigma_init=0.1, sigma_decay=0.999, sigma_min=0.01,
                 sigma_alpha=0.2, sigma_max_change=0.2,
                 alpha_init=0.01, alpha_decay=0.9999, alpha_min=0.001,
                 elite_ratio=0,
                 avg_fit_baseline=True, rank_fitness=True):
        """
        :param params_num: number of model parameters
        :param pop_size: population size
        :param sigma_init: initial STD
        :param sigma_decay: anneal STD. sigma_decay=1 --> don't anneal the STD
        :param sigma_min: stop annealing if less than this
        :param sigma_alpha: learning rate for STD
        :param sigma_max_change: clips adaptive sigma to 20%.
               restricts sigma from moving more than 20% of the original value. increases stability.
        :param alpha_init: learning rate for STD
        :param alpha_decay: annealing the learning rate. alpha_decay=1.0 --> don't anneal the learning rate
        :param alpha_min: stop annealing learning rate
        :param elite_ratio: (0,1)
        :param avg_fit_baseline: set baseline to average of batch
        :param rank_fitness: rank-based fitness-shaping - use rank rather than fitness numbers.
        """
        self.params_num = params_num
        self.pop_size = pop_size
        if self.pop_size % 2 == 0:  # Antithetic sampling + self.population[-1] == self.mu -> odd pop_size
            self.pop_size += 1
        self.rank_fitness = rank_fitness
        self.elite_ratio = elite_ratio
        self.avg_fit_baseline = avg_fit_baseline

        ####################################

        # new population's params variables:
        self.mu = np.zeros(self.params_num)

        self.sigma = np.ones(self.params_num) * sigma_init  # sigma_arr
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        # adaptive sigma params:
        self.sigma_alpha = sigma_alpha
        self.sigma_max_change = sigma_max_change

        self.epsilon_half = None
        self.epsilon = None

        ####################################

        self.alpha = alpha_init
        self.alpha_init = alpha_init
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.optimizer = optimizers.Adam(pi=self, step_size=alpha_init)  # updates self.mu

        ####################################

        self.population = None
        self.fitness_scores = None
        self.top_individual = None
        self.pop_avg_fit_history = []
        self.pop_max_fit_history = []

    def evolve(self, fitness_f):
        self.update_pop()
        self.eval_pop(fitness_f)
        self.decay_variables()

    def update_pop(self):
        # sampling epsilon (gaussian-noise vector), which constitutes the mutation:
        # antithetic_sampling:
        self.epsilon_half = np.random.randn(int(self.pop_size / 2), self.params_num) * self.sigma[None, :]
        self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half, np.zeros((1, self.params_num))])
        self.population = self.mu[None, :] + self.epsilon  # self.population[-1] == self.mu

    def eval_pop(self, fitness_function):
        self.fitness_scores = np.zeros(self.pop_size)
        for i, params in enumerate(self.population):
            self.fitness_scores[i] = fitness_function(params)

        avg_fit = np.sum(self.fitness_scores) / self.pop_size
        self.pop_avg_fit_history.append(avg_fit)

        fit_arr = compute_zero_centered_ranks(self.fitness_scores) if self.rank_fitness else self.fitness_scores

        indices_sort = np.argsort(fit_arr)[::-1]  # sorted by fitness score \ fitness rank
        top_individual_params = self.population[indices_sort[0]]
        top_individual_fitness = self.fitness_scores[indices_sort[0]]
        self.pop_max_fit_history.append(top_individual_fitness)

        if self.top_individual is None or top_individual_fitness > self.top_individual[KEY_FITNESS]:
            self.top_individual = {
                KEY_PARAMS_VEC: top_individual_params,
                KEY_FITNESS: top_individual_fitness
            }

        self.update_mu(fit_arr, indices_sort)

    def update_mu(self, fit_arr, indices_sort):
        if 0 < self.elite_ratio < 1:
            # Greedy-ES method (ignoring learning rate):
            self.mu = self.population[indices_sort[0:int(self.pop_size * self.elite_ratio)]].mean(axis=0)
        else:
            # using drift param (utilizing learning rate)
            fit_T = fit_arr[:int(self.pop_size / 2)] - fit_arr[int(self.pop_size / 2):-1]
            delta_mu = np.dot(fit_T, self.epsilon_half)

            if self.optimizer is not None:
                # the optimizer updates self.mu within itself
                self.optimizer.step_size = self.alpha
                update_ratio = self.optimizer.update(-delta_mu)  # sign flip since the optimizer is a minimizer
            else:
                self.mu += self.alpha * delta_mu  # normal SGD method?

        if self.sigma_alpha > 0:
            self.adaptive_sigma(fit_arr)

    def adaptive_sigma(self, fit_arr):
        fit_mean = (fit_arr[:int(self.pop_size / 2)] + fit_arr[int(self.pop_size / 2):-1]) / 2.0
        fit_std = 1.0 if self.rank_fitness else fit_arr.std()

        fit_baseline = np.mean(fit_arr) if self.avg_fit_baseline else fit_arr[-1]  # current mu's fitness
        fit_S = fit_mean - fit_baseline
        S = ((self.epsilon_half * self.epsilon_half - (self.sigma * self.sigma)[None, :]) / self.sigma[None, :])
        delta_sigma = (np.dot(fit_S, S)) / ((self.pop_size - 1) * fit_std)  # adaptive sigma calculation

        d_sigma = self.sigma_alpha * delta_sigma
        d_sigma = np.minimum(d_sigma, self.sigma_max_change * self.sigma)  # clip d_sigma max
        d_sigma = np.maximum(d_sigma, - self.sigma_max_change * self.sigma)  # clip d_sigma min
        self.sigma += d_sigma

    def decay_variables(self):
        if self.sigma_decay < 1:
            self.sigma[self.sigma > self.sigma_min] *= self.sigma_decay

        if self.alpha_decay < 1 and self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay
