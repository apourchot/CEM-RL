# copied and adapted from https://github.com/hardmaru/estool/blob/master/es.py
from copy import deepcopy

import numpy as np
from EA.Optimizers import Adam, SGD, BasicSGD
from pybrain.utilities import flat2triu, triu2flat
from scipy.linalg import pinv2, cholesky, inv
from scipy import outer, dot, multiply, zeros, diag, mat, sum


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))]
    which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class OpenES:

    """
    Basic Version of OpenAI Evolution Strategies
    """

    def __init__(self, num_params,
                 pop_size=100,
                 generator=None,
                 optimizer_class=Adam,
                 lr=10**-2,
                 mu_init=None,
                 mut_amp=0.1,
                 antithetic=True,
                 weight_decay=0.005,
                 rank_fitness=True):

        # misc
        self.num_params = num_params
        self.pop_size = pop_size

        # individuals
        if generator is None:
            self.individuals = [np.random.normal(
                scale=0.1, size=(pop_size, num_params))]
        else:
            self.individuals = np.array([generator() for i in range(pop_size)])
        self.new_individuals = deepcopy(self.individuals)
        self.fitness = np.zeros(pop_size)
        self.order = np.zeros(self.pop_size)
        self.to_add = None
        self.to_add_fitness = 0

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.mut_amp = mut_amp
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"

        # optimization stuff
        self.learning_rate = lr
        self.optimizer = optimizer_class(self.learning_rate)

        # other stuff
        self.rank_fitness = rank_fitness
        self.weight_decay = weight_decay

    def ask(self):
        """
        Returns a list of candidates parameters
        """
        return deepcopy(self.new_individuals)

    def add_ind(self, parameters, fitness):
        """
        Replaces the parameters of the worst individual
        """
        self.to_add = deepcopy(parameters)
        self.to_add_fitness = fitness

    def best_actor(self):
        """
        Returns the best set of parameters
        """
        return deepcopy(self.individuals[self.order[-1]])

    def best_fitness(self):
        """
        Returns the best score
        """
        return self.fitness[self.order[-1]]

    def tell(self, scores):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        # add new fitness evaluations
        self.fitness = [s for s in scores]

        # sort by fitness
        self.order = np.argsort(self.fitness)

        # replace individuals with new batch
        self.individuals = deepcopy(self.new_individuals)

        # replace worst ind with indiv to add
        if self.to_add is not None:
            self.individuals[self.order[0]] = deepcopy(self.to_add)
            self.fitness[self.order[0]] = self.to_add_fitness
            self.order = np.argsort(self.fitness)
            self.to_add = None

        # update mean
        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(
                self.weight_decay, self.individuals)
            reward += l2_decay

        epsilon = (self.individuals - self.mu) / self.mut_amp
        grad = -1. / (self.mut_amp * self.pop_size) * np.dot(reward, epsilon)

        self.optimizer.stepsize = self.learning_rate
        step = self.optimizer.step(grad)
        self.mu += step

        # sample next generation
        if self.antithetic and not self.pop_size % 2:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(self.pop_size, self.num_params)

        self.new_individuals = self.mu + epsilon * self.mut_amp

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.mut_amp ** 2)


class SNES:

    """
    Separable NES (diagonal), as described in Schaul,
    Glasmachers and Schmidhuber (GECCO'11)
    """

    def __init__(self, num_params,
                 pop_size=100,
                 generator=None,
                 optimizer_class=Adam,
                 lr=10**-2,
                 mu_init=None,
                 mut_amp=0.1,
                 antithetic=True,
                 weight_decay=0.005,
                 rank_fitness=True):

        # misc
        self.num_params = num_params
        self.pop_size = pop_size

        # individuals
        if generator is None:
            self.individuals = [np.random.normal(
                scale=0.1, size=(pop_size, num_params))]
        else:
            self.individuals = np.array([generator() for i in range(pop_size)])
        self.new_individuals = deepcopy(self.individuals)
        self.fitness = np.zeros(pop_size)
        self.order = np.zeros(self.pop_size)
        self.to_add = None
        self.to_add_fitness = 0

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.mut_amp = mut_amp
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"

        # optimization stuff
        self.learning_rate = lr
        self.optimizer = optimizer_class(self.learning_rate)

        # other stuff
        self.rank_fitness = rank_fitness
        self.weight_decay = weight_decay

    def ask(self):
        """
        Returns a list of candidates parameters
        """
        return deepcopy(self.new_individuals)

    def best_actor(self):
        """
        Returns the best set of parameters
        """
        return deepcopy(self.individuals[self.order[-1]])

    def best_fitness(self):
        """
        Returns the best score
        """
        return self.fitness[self.order[-1]]

    def add_ind(self, parameters, fitness):
        """
        Replaces the parameters of the worst individual
        """
        self.to_add = deepcopy(parameters)
        self.to_add_fitness = fitness

    def tell(self, scores):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        # add new fitness evaluations
        self.fitness = [s for s in scores]

        # sort by fitness
        self.order = np.argsort(self.fitness)

        # replace individuals with new batch
        self.individuals = deepcopy(self.new_individuals)

        # replace worst ind with indiv to add
        if self.to_add is not None:
            self.individuals[self.order[0]] = deepcopy(self.to_add)
            self.fitness[self.order[0]] = self.to_add_fitness
            self.order = np.argsort(self.fitness)
            self.to_add = None

        # update mean and sigma
        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(
                self.weight_decay, self.individuals)
            reward += l2_decay

        epsilon = (self.individuals - self.mu) / self.mut_amp
        grad = np.zeros(2 * self.num_params)
        grad[:self.num_params] = -1. / self.pop_size * \
            self.mut_amp * np.dot(reward, epsilon)
        grad[self.num_params:] = -1. / self.pop_size * 0.5 * \
            np.dot(reward, [s ** 2 - 1 for s in epsilon])

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step[:self.num_params]
        self.mut_amp = self.mut_amp * np.exp(step[self.num_params:])

        # sample next generation
        if self.antithetic and not self.pop_size % 2:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(self.pop_size, self.num_params)

        self.new_individuals = self.mu + epsilon * self.mut_amp

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.mut_amp ** 2)
