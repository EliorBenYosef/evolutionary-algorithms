"""
Optimizing: String characters (discrete-valued vector).
the String is converted into a discrete-number vector.

Evolving a population of (fixed-length) random strings toward a chosen (same-length) target string.

Running the algorithm:
* Using a 12 characters target string - should take a few minutes on a modern CPU.
* Using a longer (>12 characters) target string - will take much more time and resources to evolve.
"""

from difflib import SequenceMatcher

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.! "  # possible char values
discrete_values_num = len(alphabet)  # 26+26+4=56 characters

target = "FullyEvolved"  # 12-character string
params_num = len(target)  # target string length

max_gen_num = 100
pop_size = 1000

task_name = 'String' + str(params_num) + 'D'
optimal_fit = 1  # max similarity of SequenceMatcher


def convert_indices_to_string(individual_params):
    individual_string = ''
    for i in individual_params:
        individual_string += alphabet[i]
    return individual_string


def fitness_function(individual_params):
    """
    computes the strings' similarity
    :param individual_params: discrete-numbers vector
    :return: individual's fitness score
    """
    individual_string = convert_indices_to_string(individual_params)
    return SequenceMatcher(None, individual_string, target).ratio()
