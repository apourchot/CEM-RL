import numpy as np
import random

from copy import deepcopy
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


class GA:
    """
    Basic population based genetic algorithm
    """

    def __init__(self, num_params,
                 pop_size=100,
                 elite_frac=0.1,
                 mut_rate=0.9,
                 mut_amp=0.1,
                 generator=None):

        # misc
        self.num_params = num_params
        self.pop_size = pop_size
        self.n_elites = int(self.pop_size * elite_frac)

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

        # mutations
        self.mut_amp = mut_amp
        self.mut_rate = mut_rate

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

    def set_new_params(self, new_params):
        """
        Replaces the current new_population with the
        given population of parameters
        """
        self.new_individuals = deepcopy(np.array(new_params))

    def ask(self):
        """
        Returns the newly created individual(s)
        """
        return deepcopy(self.new_individuals)

    def tell(self, scores):
        """
        Updates the population
        """
        assert(len(scores) == len(self.new_individuals)
               ), "Inconsistent reward_table size reported."

        # add new fitness evaluations
        self.fitness = [s for s in scores]

        # sort by fitness
        self.order = np.argsort(self.fitness)

        # replace individuals with new batch
        self.individuals = deepcopy(self.new_individuals)

        # replace worst ind with ind to add
        if self.to_add is not None:
            self.individuals[self.order[0]] = deepcopy(self.to_add)
            self.fitness[self.order[0]] = self.to_add_fitness
            self.order = np.argsort(self.fitness)
            self.to_add = None

        # tournament selection
        tmp_individuals = []
        while len(tmp_individuals) < (self.pop_size - self.n_elites):
            k, l = np.random.choice(range(self.pop_size), 2, replace=True)
            if self.fitness[k] > self.fitness[l]:
                tmp_individuals.append(deepcopy(self.individuals[k]))
            else:
                tmp_individuals.append(deepcopy(self.individuals[l]))

        # mutation
        tmp_individuals = np.array(tmp_individuals)
        for ind in range(tmp_individuals.shape[0]):
            u = np.random.rand(self.num_params)
            params = tmp_individuals[ind]
            noise = np.random.normal(
                loc=1, scale=self.mut_amp * (u < self.mut_rate))
            params *= noise

        # new population
        self.new_individuals[self.order[:self.pop_size -
                                        self.n_elites]] = np.array(tmp_individuals)
