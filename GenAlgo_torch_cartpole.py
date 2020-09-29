"""
Optimizing a policy NN (state --> probability distribution over actions) in the CartPole environment.
"""

import numpy as np
import torch
import torch.distributions
import torch.nn.functional
import gym
from matplotlib import pyplot as plt
from os import path, mkdir

#########################################

# adjustable variables:

hidden_layers_units = [25, 10]  # The individual NN hidden layers

gen_num = 25
pop_size = 500
mut_rate = 1e-2  # 1e-3  # the % of elements to mutate in each individual
tournament_size = 0.2  # 20% of the population

env_name = 'CartPole-v0'
input_dims = 4  # input layer
n_actions = 2  # output layer

#########################################

# constant variables:

params_vec_key = 'params_vec'
fitness_key = 'fitness'

layers_weights_shapes = []  # the parameter vector's layers (each layer's weight matrix dimension)
params_num = 0  # the total number parameters
for i in range(len(hidden_layers_units) + 1):
    v_curr = n_actions if i == len(hidden_layers_units) else hidden_layers_units[i]
    v_prev = input_dims if i == 0 else hidden_layers_units[i - 1]
    layers_weights_shapes.append((v_curr, v_prev))
    params_num += (v_curr * (v_prev + 1))

#########################################


def init_population():
    """
    creates a population of agents (individuals),
    each is a dict that stores its NN parameters vector & fitness score
    """
    population = []
    for i in range(pop_size):
        population.append({params_vec_key: torch.randn(params_num) / 2.0, fitness_key: 0})
    return population


def split_params_vec(params_vec):
    """
    :param params_vec: NN parameters vector
    :return: params_by_layer: a list of layer-wise tuples: (layer's weight matrix, layer's bias vector)
    """
    params_by_layer = []
    end_pt = 0
    for i, layer in enumerate(layers_weights_shapes):
        start_pt, end_pt = end_pt, end_pt + np.prod(layer)
        weights = params_vec[start_pt:end_pt].view(layer)
        start_pt, end_pt = end_pt, end_pt + layer[0]
        bias = params_vec[start_pt:end_pt]
        params_by_layer.append((weights, bias))
    return params_by_layer


def construct_model_and_pass_state(s, params_by_layer):
    """
    constructs the NN model, and passes the input state into it
    :param s: input state
    :param params_by_layer: a list of layer-wise tuples: (layer's weight matrix, layer's bias vector)
    :return: probabilities over actions
    """
    for i, layer_params in enumerate(params_by_layer):
        w, b = layer_params
        x = torch.nn.functional.linear(s if i == 0 else x, w, b)  # logits
        x = torch.relu(x) if i != len(params_by_layer) - 1 else torch.softmax(x, dim=0)  # probs
    return x


def calc_fitness(individual):
    """
    testing the model
    assessing the individual's fitness by testing it in the CartPole environment
    (until it loses the game and returns the number of time steps it lasted as its fitness score).
    :return:
    """
    params_by_layer = split_params_vec(individual[params_vec_key])

    done = False
    fitness_score = 0
    s = torch.from_numpy(env.reset()).float()
    while not done:
        probs = construct_model_and_pass_state(s, params_by_layer)
        a = torch.distributions.Categorical(probs=probs).sample().item()
        s_, r, done, info = env.step(a)
        s = torch.from_numpy(s_).float()
        fitness_score += r
    return fitness_score


def eval_population(population):
    """
    evaluates all individuals and updates their fitness score
    :return: update population (with fitness scores), average population fitness
    """
    fit_sum = 0
    for individual in population:
        fit = calc_fitness(individual)
        individual[fitness_key] = fit
        fit_sum += fit
    return population, fit_sum / pop_size


def crossover(p1, p2):
    """
    Parents crossover (AKA genetic recombination)
    parents (p) --> 2 offspring (o)
    """
    cross_pt = np.random.randint(params_num)

    o1_params = torch.zeros(params_num)
    o1_params[0:cross_pt] = p1[params_vec_key][:cross_pt]
    o1_params[cross_pt:] = p2[params_vec_key][cross_pt:]
    o1 = {params_vec_key: o1_params, fitness_key: 0.0}

    o2_params = torch.zeros(params_num)
    o2_params[0:cross_pt] = p2[params_vec_key][:cross_pt]
    o2_params[cross_pt:] = p1[params_vec_key][cross_pt:]
    o2 = {params_vec_key: o2_params, fitness_key: 0.0}

    return o1, o2


def mutate(individual):
    """
    Mutating an individual's parameter vector, by randomly changing a few elements of the parameter vector.
    The mutation rate controls the number of changed elements.
    :return: mutated individual
    """
    mut_num = int(mut_rate * params_num)  # number of elements to mutate
    mut_indices = np.random.randint(low=0, high=params_num, size=(mut_num,))
    # torch.randn() returns a tensor filled with random numbers from a standard normal distribution (mean 0, variance 1)
    individual[params_vec_key][mut_indices] = torch.randn(mut_num) / 10.0  # TODO: remove "/ 10.0" ?
    return individual


def next_gen(population):
    """
    creating the next generation.
    Tournament Selection of parents. the batch is the subset of the population.
    """
    new_pop = []
    while len(new_pop) < pop_size:
        rand_indices = np.random.randint(low=0, high=pop_size, size=(int(tournament_size * pop_size)))
        batch = np.array([[index, individual[fitness_key]] for (index, individual) in enumerate(population)
                          if index in rand_indices])
        batch_sort = batch[batch[:, 1].argsort()]  # sort by fitness scores in ascending order
        top_0_i, top_1_i = int(batch_sort[-1][0]), int(batch_sort[-2][0])  # get the top (last) two individuals indices
        top_0_parent, top_1_parent = population[top_0_i], population[top_1_i]  # get the individuals
        offspring = crossover(top_0_parent, top_1_parent)
        mut_offspring = [mutate(offspring[0]), mutate(offspring[1])]
        new_pop.extend(mut_offspring)
    return new_pop


#########################################

env = gym.make(env_name)
pop_avg_fit_history = []
pop = init_population()
for i in range(gen_num):
    if i != 0:
        pop = next_gen(pop)
    pop, avg_fit = eval_population(pop)
    pop_avg_fit_history.append(avg_fit)

plt.title('CartPole - Population Average Fitness')
plt.ylabel('Fitness')
plt.xlabel('Generation')
x = [i + 1 for i in range(gen_num)]
plt.plot(x, pop_avg_fit_history)
if not path.isdir('results'):
    mkdir('results')
plt.savefig('results/population_average_fitness_cartpole.png')
plt.show()
plt.close()

pop.sort(key=lambda individual: individual[fitness_key], reverse=True)  # descending fitness score
print(f"Best individual's fitness: {str(pop[0][fitness_key])}")  # print the highest ranked individual in the population
