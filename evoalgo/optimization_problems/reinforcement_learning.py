"""
Optimizing a policy NN (state --> probability distribution over actions) in the CartPole environment.
Optimizing: NN params (weights and biases).
"""

import numpy as np
import torch
import torch.distributions
import torch.nn.functional
import gym

max_gen_num = 25
pop_size = 500

env_name = 'CartPole-v0'
input_dims = 4  # input layer
n_actions = 2  # output layer
optimal_fit = None

hidden_layers_units = [25, 10]  # The individual NN hidden layers

#########################################

# NN-model specific

layers_weights_shapes = []  # the parameter vector's layers (each layer's weight matrix dimension)
params_num = 0  # the total number parameters
for i in range(len(hidden_layers_units) + 1):
    v_curr = n_actions if i == len(hidden_layers_units) else hidden_layers_units[i]
    v_prev = input_dims if i == 0 else hidden_layers_units[i - 1]
    layers_weights_shapes.append((v_curr, v_prev))
    params_num += (v_curr * (v_prev + 1))
task_name = 'CartPole' + str(params_num) + 'D'


def split_model_params_vec(params_vec):
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


#########################################

# RL specific

env = gym.make(env_name)


def fitness_function(individual_params):
    """
    assessing the individual's fitness by testing its model in the CartPole environment
    (until it loses the game and returns the number of time steps it lasted as its fitness score).
    :return: individual's fitness score
    """
    if isinstance(individual_params, np.ndarray):
        individual_params = torch.as_tensor(individual_params, dtype=torch.float32)

    params_by_layer = split_model_params_vec(individual_params)

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
