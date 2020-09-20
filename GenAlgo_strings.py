"""
Evolving a population of (fixed-length) random strings toward a chosen (same-length) target string.

Running the algorithm:
* Using a 12 characters target string - should take a few minutes on a modern CPU.
* Using a longer (>12 characters) target string - will take much more time and resources to evolve.
"""

import random
from difflib import SequenceMatcher
from matplotlib import pyplot as plt
from os import path, mkdir

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.! "
target = "FullyEvolved"  # 12-character string
individuals_that_reached_target = 0

str_len = len(target)
max_gen_num = 150
pop_size = 900
mut_rate = 1e-5

#########################################


class Individual:
    def __init__(self, string, fitness=0):
        self.string = string
        self.fitness = fitness  # fitness score


def init_population():
    population = []
    for i in range(pop_size):
        population.append(Individual(''.join(random.choices(alphabet, k=str_len))))
    return population


def calc_fitness(individual):
    """
    the fitness function
    :return: individual's fitness score
    """
    global individuals_that_reached_target
    strings_similarity = SequenceMatcher(None, individual, target).ratio()
    if strings_similarity == 1:
        individuals_that_reached_target += 1
    return strings_similarity


def eval_population(population):
    """
    evaluates all individuals and updates their fitness score
    :return: update population (with fitness scores), average population fitness
    """
    fit_sum = 0
    for individual in population:
        fit = calc_fitness(individual.string)
        individual.fitness = fit
        fit_sum += fit
    return population, fit_sum / pop_size


def crossover(p1, p2):
    """
    Parents crossover (AKA genetic recombination)
    parents (p) --> 2 offspring (o)
    """
    cross_pt = random.randint(0, str_len)
    o1 = Individual(p1.string[:cross_pt] + p2.string[cross_pt:])
    o2 = Individual(p2.string[:cross_pt] + p1.string[cross_pt:])
    return o1, o2


def mutate(individual):
    """
    randomly flipping a few characters in the string (replacing it with a random character)
    :return: mutated individual
    """
    mut_individual = ''
    for char in individual.string:
        if random.random() < mut_rate:
            mut_individual += random.choices(alphabet, k=1)[0]
        else:
            mut_individual += char
    return Individual(mut_individual)


def next_gen(population):
    """
    creating the next generation.
    Fitness-Proportionate Selection (FPS) of parents.
    """
    new_pop = []
    while len(new_pop) < pop_size:
        parents = random.choices(population, k=2, weights=[individual.fitness for individual in population])
        offspring = crossover(parents[0], parents[1])
        mut_offspring = [mutate(offspring[0]), mutate(offspring[1])]
        new_pop.extend(mut_offspring)
    return new_pop


#########################################

pop_avg_fit_history = []
pop = init_population()
gen = 0
for gen in range(max_gen_num):  # iterate for a fixed number of generations.
    if gen != 0:
        next_gen(pop)
    pop, avg_fit = eval_population(pop)
    pop_avg_fit_history.append(avg_fit)
    if individuals_that_reached_target > 0:  # stop the algorithm when target is reached.
        print(f'Target reached at Gen {str(gen + 1)}')
        break

plt.title('Strings - Population Average Fitness')
plt.ylabel('Fitness')
plt.xlabel('Generations')
x = [i + 1 for i in range(gen + 1)]
plt.plot(x, pop_avg_fit_history)
if not path.isdir('results'):
    mkdir('results')
plt.savefig('results/population_average_fitness_strings.png')
plt.show()
plt.close()

if max_gen_num != 0:
    pop.sort(key=lambda individual: individual.fitness, reverse=True)  # descending fitness score
    print(f'Best individual: {pop[0].string}')  # print the highest ranked individual in the population
