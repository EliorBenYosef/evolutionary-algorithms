"""
taken from: https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
"""

import numpy as np


class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.params_num
        self.epsilon = epsilon
        self.t = 0

    def update(self, global_g):
        self.t += 1
        step = self._compute_step(global_g)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class BasicSGD(Optimizer):
    def __init__(self, pi, step_size):
        Optimizer.__init__(self, pi)
        self.step_size = step_size

    def _compute_step(self, global_g):
        step = -self.step_size * global_g
        return step


class SGD(Optimizer):
    def __init__(self, pi, step_size, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.step_size, self.momentum = step_size, momentum

    def _compute_step(self, global_g):
        self.v = self.momentum * self.v + (1. - self.momentum) * global_g
        step = -self.step_size * self.v
        return step


class Adam(Optimizer):
    def __init__(self, pi, step_size, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, global_g):
        a = self.step_size * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * global_g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (global_g * global_g)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
