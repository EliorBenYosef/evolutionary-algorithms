# Evolutionary Algorithms implementations
Evolutionary Algorithms implementations, for various (discrete & continuous) optimization problems.

### TL;DR

**Evolutionary Algorithms (EAs):**
* [Genetic Algorithm (GA)](../master/evoalgo/evolutionary_algorithms/genetic_algorithm.py)
  * using various Selection, Crossover & Mutation strategies.
* [Evolution Strategy (ES)](../master/evoalgo/evolutionary_algorithms/evolution_strategy.py)
  * PEPG
  * OpenAI-ES
  * CMA-ES

**Optimization problems (of increasing difficulty):**
* [String](../master/evoalgo/optimization_problems/string_opt.py) - 
optimizing a fixed-length String (discrete-valued vector) towards a chosen target String of 12 characters.
  * For generality purposes, the String (individual & target) is converted into a discrete-number vector (and vice-versa), 
  which is optimized.
* [Rastrigin function](../master/evoalgo/optimization_problems/rastrigin.py) -
optimizing the Rastrigin function's input parameters (x).
* [Policy Neural-Network](../master/evoalgo/optimization_problems/policy_nn.py) 
**for autonomous agent control** - 
optimizing the Policy Neural-Network's parameters (layers' weights and biases)
  * A Policy network receives the **environment state** as an input, 
  and outputs a **probability distribution over all possible actions**.
  * Any AI Gym environment can be chosen, as long as the relevant variables parameters
  (`env_name`, `input_dims`, `n_actions`, `optimal_fit`) are changed accordingly.
  Here, AI Gym's **CartPole** environment is chosen as an example.

Optimization problem | Type | Parameters number
--- | --- | ---
**String** | Discrete-number vector | 12
**Rastrigin function** | Real-number vector | 100
**Policy Neural Network for autonomous agent control** | Real-number vector | 407

the higher the dimensionality (parameters number) of the optimization problem, the harder it is, 
and the slower the optimization process is.

### Table of contents:

* [Intro](https://github.com/EliorBenYosef/evolutionary-algorithms#intro)
  * [Evolutionary Algorithms](https://github.com/EliorBenYosef/evolutionary-algorithms#evolutionary-algorithms)
  * [Optimization Problems](https://github.com/EliorBenYosef/evolutionary-algorithms#optimization-problems)
* [Results](https://github.com/EliorBenYosef/evolutionary-algorithms#results)
  * [Evolutionary Algorithms Comparison](https://github.com/EliorBenYosef/evolutionary-algorithms#evolutionary-algorithms-comparison)
  * [Genetic Algorithms Comparison](https://github.com/EliorBenYosef/evolutionary-algorithms#genetic-algorithms-comparison)
* [How to use](https://github.com/EliorBenYosef/evolutionary-algorithms#how-to-use)

## Intro

###  Evolutionary Algorithms
black-box optimization via evolution.
Evolving a population of fixed-length vector toward a chosen (same-length) target string.
starts random-valued (starts with random values)

<p align="center">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/EvoAlgo_bear.jpeg" width="400">
</p>

####  Genetic Algorithm

####  Evolution Strategy

###  Optimization Problems


## Results
optimization process results:

## Evolutionary Algorithms Comparison

Legend:
<p align="center">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ea_comparison/Legend.png" width="400">
</p>


#### Rastrigin function (100 input parameters)

<p align="left">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ea_comparison/Rastrigin100D-Pop100-Avg.png" width="400">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ea_comparison/Rastrigin100D-Pop100-Max.png" width="400">
</p>

#### Policy Neural-Network (407 parameters)
Environment: CartPole.
layers: 4-25-10-2.

<p align="left">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ea_comparison/CartPole407D-Pop500-Avg.png" width="400">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ea_comparison/CartPole407D-Pop500-Max.png" width="400">
</p>

## Genetic Algorithms Comparison

Legend:
<p align="center">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/Legend.png" width="800">
</p>

#### String (12 characters)
Notice how the optimization is better with a larger population size.

<p align="left">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/String12D-Pop1000-Avg.png" width="400">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/String12D-Pop1000-Max.png" width="400">
</p>

<p align="left">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/String12D-Pop100-Avg.png" width="400">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/String12D-Pop100-Max.png" width="400">
</p>

#### Rastrigin function (100 input parameters)
Notice how the optimization is better with a higher (and constant) mutation sigma.

**Sigma = 1**
<p align="left">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/Rastrigin100D-Pop100-Avg-Sigma1.png" width="400">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/Rastrigin100D-Pop100-Max-Sigma1.png" width="400">
</p>

**Sigma = 0.5**
<p align="left">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/Rastrigin100D-Pop100-Avg-Sigma0.5.png" width="400">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/Rastrigin100D-Pop100-Max-Sigma0.5.png" width="400">
</p>

**Sigma: Init = 0.5, Decay = 0.9, Min = 0.01**
<p align="left">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/Rastrigin100D-Pop100-Avg-Sigma0.5_Min0.01_Decay0.9.png" width="400">
  <img src="https://github.com/EliorBenYosef/evolutionary-algorithms/blob/master/images/results/ga_comparison/Rastrigin100D-Pop100-Max-Sigma0.5_Min0.01_Decay0.9.png" width="400">
</p>

## How to use

Examples of how to use:
* [compare_EAs.py](../master/compare_EAs.py)
* [compare_GAs.py](../master/compare_GAs.py)

