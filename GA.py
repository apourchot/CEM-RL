import numpy as np
import random

from copy import deepcopy


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
        self.individuals = np.array([generator() for i in range(pop_size)])
        self.fitness = np.zeros(pop_size)
        self.order = np.zeros(self.pop_size, dtype=np.int64)
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

    def best_index(self):
        """
        Returns the index of the best set of parameters
        """
        return self.order[-1]

    def best_fitness(self):
        """
        Returns the best score
        """
        return self.fitness[self.order[-1]]

    def add(self, parameters, fitness):
        """
        Replaces the parameters of the worst individual
        """
        index = self.order[0]
        if fitness < self.fitness[index]:
            return
        self.individuals[index] = deepcopy(parameters)
        self.fitness[index] = fitness
        self.order = np.argsort(self.fitness)

    def set_new_params(self, new_params):
        """
        Replaces the current new_population with the
        given population of parameters
        """
        self.individuals = deepcopy(np.array(new_params))

    def ask(self):
        """
        Returns the newly created individual(s)
        """
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

        # replace individuals with new batch
        self.individuals[self.order[:self.pop_size -
                                    self.n_elites]] = np.array(tmp_individuals)

        return deepcopy(self.individuals)

    def tell(self, solutions, scores):
        """
        Updates the population
        """
        assert(len(scores) == len(self.individuals)
               ), "Inconsistent reward_table size reported."

        # add new fitness evaluations
        self.fitness = [s for s in scores]

        # sort by fitness
        self.order = np.argsort(self.fitness)
