# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
import numpy as np


class Optimizer(object):

    def __init__(self, epsilon=1e-08):
        self.epsilon = epsilon

    def step(self, grad):
        raise NotImplementedError


class BasicSGD(Optimizer):
    """
    Standard gradient descent
    """

    def __init__(self, stepsize):
        Optimizer.__init__(self)
        self.stepsize = stepsize

    def step(self, grad):
        step = -self.stepsize * grad
        return step


class SGD(Optimizer):
    """
    Gradient descent with momentum
    """

    def __init__(self, stepsize, momentum=0.9):
        Optimizer.__init__(self)
        self.stepsize, self.momentum = stepsize, momentum

    def step(self, grad):
        if not hasattr(self, 'v'):
            self.v = np.zeros(grad.shape[0], dtype=np.float32)
        self.v = self.momentum * self.v + (1. - self.momentum) * grad
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    """
    Adam optimizer
    """

    def __init__(self, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def step(self, grad):

        if not hasattr(self, "m"):
            self.m = np.zeros(grad.shape[0], dtype=np.float32)
        if not hasattr(self, "v"):
            self.v = np.zeros(grad.shape[0], dtype=np.float32)

        self.t += 1
        a = self.stepsize * np.sqrt(1 - self.beta2 **
                                    self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)

        return step
