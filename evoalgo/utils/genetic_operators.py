"""
Genetic Operators.
"""

import random
import numpy as np
import torch
from evoalgo.const import KEY_PARAMS_VEC, KEY_FITNESS


class Selection:
    """
    p - parent
    o - offspring
    """

    @staticmethod
    def fitness_proportionate(population):
        """
        Fitness-Proportionate Selection (FPS) of parents. Stochastic selection method.
        """
        # np approach - works only with positive weights, problematic with negative weights.
        # p = np.array([individual[KEY_FITNESS] for individual in population])
        # p /= p.sum()  # normalize. problem with negative weights -> sum = 0
        # parents = np.random.choice(population, size=2, p=p)

        # a single pointer, on a wheel that is spun 2 times:
        parents = random.choices(population, k=2, weights=[individual[KEY_FITNESS] for individual in population])

        p1_params = parents[0][KEY_PARAMS_VEC]
        p2_params = parents[1][KEY_PARAMS_VEC]
        return p1_params, p2_params

    @staticmethod
    def stochastic_top_sampling(population_sort, top_size=0.1):
        """
        Stochastic Top Sampling (STS) Selection of parents. Stochastic selection method.
        selecting from the top T (% of the) individuals.
        :param population_sort: sorted population by descending fitness-score.
        :param top_size: top ratio. default: 0.1 -> top 10% of the population.
        """
        pop_size = len(population_sort)
        rand_elite_pair_indices = np.random.randint(low=0, high=(int(top_size * pop_size)), size=2)

        p1_params = population_sort[rand_elite_pair_indices[0]][KEY_PARAMS_VEC]
        p2_params = population_sort[rand_elite_pair_indices[1]][KEY_PARAMS_VEC]
        return p1_params, p2_params

    @staticmethod
    def tournament(population, tournament_size=0.2):
        """
        Tournament Selection of parents. Semi-deterministic selection method.
        the batch is the tournament batch (a subset of the population).
        :param tournament_size: tournament ratio. default: 0.2 -> random 20% of the population.
        """
        pop_size = len(population)
        rand_indices = np.random.randint(low=0, high=pop_size, size=(int(tournament_size * pop_size)))
        batch = np.array([[i, individual[KEY_FITNESS]] for (i, individual) in enumerate(population)
                          if i in rand_indices])
        batch_sort = batch[batch[:, 1].argsort()]  # sort by ascending fitness-score

        p1_params = population[int(batch_sort[-1, 0])][KEY_PARAMS_VEC]  # top_0_parent params
        p2_params = population[int(batch_sort[-2, 0])][KEY_PARAMS_VEC]  # top_1_parent params
        return p1_params, p2_params

    ######################################

    # stochastic_universal_sampling is not operational yet... (there's a TODO)
    @staticmethod
    def stochastic_universal_sampling(population_sort):
        """
        Stochastic Universal Sampling (SUS) of parents. Stochastic selection method.
        provides no bias and minimal spread.
        :param population_sort: sorted population by descending fitness-score.
        :return: a list of parameters of N parents  # TODO: to be randomly? paired (non-identical individuals), recombined and mutated.
        """
        # adjust negative weights (crucial for the sum later)
        min_fit = np.array([individual[KEY_FITNESS] for individual in population_sort]).min()
        if min_fit < 0:
            for individual in population_sort:
                individual[KEY_FITNESS] -= min_fit

        total_fit = sum([individual[KEY_FITNESS] for individual in population_sort])
        if total_fit == 0:  # meaning: each individual's fittness is 0
            for individual in population_sort:
                individual[KEY_FITNESS] += 1

        wheel = []
        fit_limit = 0.0
        for i, individual in enumerate(population_sort):
            fit_limit += individual[KEY_FITNESS]
            wheel.append((fit_limit, i))

        # N equally-spaced pointers, on a wheel that is spun once (single sampling):
        pop_size = len(population_sort)
        pointer_size = 0
        while pointer_size == 0:
            pointer_size = random.uniform(0, total_fit/pop_size)

        individuals_indices = []
        curr_i = 0
        current_pointer = pointer_size
        for fit_limit, i in wheel:
            while current_pointer < fit_limit and curr_i < pop_size:
                individuals_indices.append(i)
                curr_i += 1
                current_pointer = (curr_i + 1) * pointer_size

        p_params_list = [population_sort[i][KEY_PARAMS_VEC] for i in individuals_indices]

        return p_params_list

    # truncation is not operational yet... (there's a TODO)
    @staticmethod
    def truncation(population_sort, truncation_size=0.1):
        """
        Truncation Selection of parents. Deterministic selection method.
        selecting the top T (% of the) individuals.
        :param population_sort: sorted population by descending fitness-score.
        :param truncation_size: truncation ratio. default: 0.1 -> top 10% of the population.
        :return: a list of parameters of (N * truncation_size) parents  # TODO: to be randomly? paired (non-identical individuals), recombined and mutated.
        """
        pop_size = len(population_sort)
        p_params_list = [individual[KEY_PARAMS_VEC] for individual in population_sort[:int(truncation_size * pop_size)]]

        return p_params_list

    @staticmethod
    def elitism(population_sort, elite_size=0.1):
        """
        Elitism Selection of individuals. Deterministic selection method.
        :param population_sort: sorted population by descending fitness-score.
        :param elite_size: elite ratio. default: 0.1 -> top 10% of the population.
        :return: a list of individuals, to be directly copied to the next generation's (new) population.
        """
        pop_size = len(population_sort)
        individuals_list = []
        for individual in population_sort[:int(elite_size * pop_size)]:
            individuals_list.append({KEY_PARAMS_VEC: individual[KEY_PARAMS_VEC], KEY_FITNESS: None})
        return individuals_list


class Crossover:
    """
    Parents crossover (AKA genetic recombination)
    2 parents (p) --> 2 offspring (o)
    """

    @staticmethod
    def single_pt(p1, p2, params_num, is_torch):
        """
        Positional method.
        :param p1: p1_params
        :param p2: p2_params
        :param params_num: vector's length
        :return: o1_params, o2_params
        """
        cross_pt = np.random.randint(low=1, high=params_num)

        o1 = torch.clone(p1) if is_torch else np.copy(p1)
        o1[cross_pt:] = p2[cross_pt:]

        o2 = torch.clone(p2) if is_torch else np.copy(p2)
        o2[cross_pt:] = p1[cross_pt:]

        return o1, o2

    @staticmethod
    def two_pt(p1, p2, params_num, is_torch):
        """
        Positional method.
        :param p1: p1_params
        :param p2: p2_params
        :param params_num: vector's length
        :return: o1_params, o2_params
        """
        cross_pt_1, cross_pt_2, cross_pt_low, cross_pt_high = 0, 0, 0, 0
        while cross_pt_1 == cross_pt_2:
            cross_pt_1, cross_pt_2 = np.random.randint(low=1, high=params_num, size=2)
            if cross_pt_1 != cross_pt_2:
                if cross_pt_1 < cross_pt_2:
                    cross_pt_low, cross_pt_high = cross_pt_1, cross_pt_2
                else:
                    cross_pt_low, cross_pt_high = cross_pt_2, cross_pt_1

        o1 = torch.clone(p1) if is_torch else np.copy(p1)
        o1[cross_pt_low:cross_pt_high] = p2[cross_pt_low:cross_pt_high]

        o2 = torch.clone(p2) if is_torch else np.copy(p2)
        o2[cross_pt_low:cross_pt_high] = p1[cross_pt_low:cross_pt_high]

        return o1, o2

    @staticmethod
    def uniform(p1, p2, params_num, is_torch):
        """
        Non-positional method.
        :param p1: p1_params
        :param p2: p2_params
        :param params_num: vector's length
        :return: o1_params, o2_params
        """
        indices = np.where(np.random.rand(params_num) > 0.5)

        o1 = torch.clone(p1) if is_torch else np.copy(p1)
        o1[indices] = p2[indices]

        o2 = torch.clone(p2) if is_torch else np.copy(p2)
        o2[indices] = p1[indices]

        return o1, o2


class Mutation:
    """
    Mutating an individual's parameter vector
    """

    @staticmethod
    def deterministic(individual, params_num, discrete_values_num, is_torch, mut_rate=1e-2):
        """
        mutation by randomly changing a proportional number elements of the parameter vector.
        recommended for a high number of params (long vectors - NN params).
        :param individual: individual_params
        :param params_num: vector's length
        :param mut_rate: the % of elements to mutate in each individual. determines the number of changed elements.
        :return: individual's mutated params
        """
        mut_num = int(mut_rate * params_num)  # number of elements to mutate

        if mut_num > 0:
            mut_indices = np.random.randint(low=0, high=params_num, size=(mut_num,))
            if discrete_values_num is not None:
                # randomly flipping values (replacing them with random values)
                individual[mut_indices] = \
                    torch.randint(low=0, high=discrete_values_num, size=(mut_num,)) if is_torch else \
                    np.random.randint(low=0, high=discrete_values_num, size=mut_num)
            else:
                # sample a random number from a standard normal distribution (mean 0, variance 1)
                # the division gives a number which is close to 0 -> good for the NN weights.
                individual[mut_indices] = (torch.randn(mut_num) if is_torch else np.random.randn(mut_num)) / 10.0  # TODO: test with 2.0, 10.0, and without

        return individual

    @staticmethod
    def stochastic_uniform(individual, params_num, discrete_values_num, is_torch, mut_rate=1e-2):
        """
        mutation by randomly changing a random number elements of the parameter vector.
        recommended for a small number of params (short vectors).
        :param individual: individual_params
        :param params_num: vector's length
        :param mut_rate: the chance of changing each single element.
        :return: individual's mutated params
        """
        for i in range(params_num):
            if random.random() < mut_rate:
                if discrete_values_num is not None:
                    # randomly flipping values (replacing them with random values)
                    individual[i] = \
                        torch.randint(low=0, high=discrete_values_num, size=(1,)) if is_torch else \
                        np.random.randint(low=0, high=discrete_values_num)
                else:
                    # sample a random number from a standard normal distribution (mean 0, variance 1)
                    # the division gives a number which is close to 0 -> good for the NN weights.
                    individual[i] = (torch.randn(1) if is_torch else np.random.randn()) / 10.0  # TODO: test with 2.0, 10.0, and without

        return individual

    @staticmethod
    def gaussian_noise(individual, params_num, discrete_values_num, is_torch, sigma=0.5):
        """
        Mutation by adding a random noise vector (epsilon),
        which is sampled from a "standard normal" distribution.
        recommended for when values distance (from each other) is meaningful.
        :param individual: individual_params
        :param params_num: vector's length
        :param sigma:
        :return: individual's mutated params
        """
        # sample a random number from a standard normal distribution (mean 0, variance 1)
        epsilon = (torch.randn(params_num) if is_torch else np.random.randn(params_num)) * sigma
        if discrete_values_num is not None:
            epsilon_disctere = torch.zeros(params_num, dtype=torch.int32) if is_torch else np.zeros(params_num, dtype=np.int32)
            for i in range(params_num):
                epsilon_disctere[i] = int(round(epsilon[i]))
            epsilon = epsilon_disctere

        individual += epsilon

        if discrete_values_num is not None:
            for i in range(params_num):
                if individual[i] < 0:
                    individual[i] = 0
                elif individual[i] > discrete_values_num - 1:
                    individual[i] = discrete_values_num - 1

        return individual
