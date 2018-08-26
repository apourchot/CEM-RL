import numpy as np
from copy import deepcopy

from Optimizers import Adam, BasicSGD


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


class VES:

    """
    Basic Version of OpenAI Evolution Strategies
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=0.1,
                 lr=10**-2,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0,
                 rank_fitness=True):

        # misc
        self.num_params = num_params
        self.first_interation = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init

        # optimization stuff
        self.learning_rate = lr
        self.optimizer = Adam(self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self):
        """
        Returns a list of candidates parameterss
        """
        if self.antithetic:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(self.pop_size, self.num_params)

        return self.mu + epsilon * self.sigma

    def tell(self, scores, solutions):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        epsilon = (solutions - self.mu) / self.sigma
        grad = -1/(self.sigma * self.pop_size) * np.dot(reward, epsilon)

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)


class GES:

    """
    Guided Evolution Strategies
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=0.1,
                 lr=10**-2,
                 alpha=0.5,
                 beta=2,
                 k=1,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0,
                 rank_fitness=False):

        # misc
        self.num_params = num_params
        self.first_interation = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.U = np.ones((self.num_params, k))

        # optimization stuff
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.learning_rate = lr
        self.optimizer = Adam(self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self):
        """
        Returns a list of candidates parameterss
        """
        if self.antithetic:
            epsilon_half = np.sqrt(self.alpha / self.num_params) * \
                np.random.randn(self.pop_size // 2, self.num_params)
            epsilon_half += np.sqrt((1 - self.alpha) / self.k) * \
                np.random.randn(self.pop_size // 2, self.k) @ self.U.T
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.sqrt(self.alpha / self.num_params) * \
                np.random.randn(self.pop_size, self.num_params)
            epsilon += np.sqrt(1 - self.alpha) * \
                np.random.randn(self.pop_size, self.num_params) @ self.U.T

        return self.mu + epsilon * self.sigma

    def tell(self, scores, solutions):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        epsilon = (solutions - self.mu) / self.sigma
        grad = -self.beta/(self.sigma * self.pop_size) * \
            np.dot(reward, epsilon)

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step

    def add(self, params, grads, fitness):
        """
        Adds new "gradient" to U
        """
        if params is not None:
            self.mu = params
        grads = grads / np.linalg.norm(grads)
        self.U[:, -1] = grads

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)


class sepCMAES:

    """
    CMAES implementation adapted from
    https://en.wikipedia.org/wiki/CMA-ES#Example_code_in_MATLAB/Octave
    """

    def __init__(self,
                 num_params,
                 mu_init=None,
                 sigma_init=1,
                 step_size_init=1,
                 pop_size=255,
                 antithetic=False,
                 weight_decay=0.01,
                 rank_fitness=True):

        # distribution parameters
        self.num_params = num_params
        if mu_init is not None:
            self.mu = np.array(mu_init)
        else:
            self.mu = np.zeros(num_params)
        self.antithetic = antithetic

        # stuff
        self.step_size = step_size_init
        self.p_c = np.zeros(self.num_params)
        self.p_s = np.zeros(self.num_params)
        self.cov = sigma_init ** 2 * np.ones(num_params)

        # selection parameters
        self.pop_size = pop_size
        self.parents = pop_size // 2
        self.weights = np.array([np.log((self.parents + 0.5) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()
        self.parents_eff = 1 / (self.weights ** 2).sum()
        self.rank_fitness = rank_fitness
        self.weight_decay = weight_decay

        # adaptation  parameters
        self.c_s = (self.parents_eff + 2) / \
            (self.num_params + self.parents_eff + 3)
        self.c_c = 4 / (self.num_params + 4)
        self.c_cov = 1 / self.parents_eff * 2 / ((self.num_params + np.sqrt(2)) ** 2) + (1 - 1 / self.parents_eff) * \
            min(1, (2 * self.parents_eff - 1) /
                (self.parents_eff + (self.num_params + 2) ** 2))
        self.c_cov *= (self.num_params + 2) / 3
        self.d_s = 1 + 2 * \
            max(0, np.sqrt((self.parents_eff - 1) /
                           (self.num_params + 1) - 1)) + self.c_s
        self.chi = np.sqrt(self.num_params) * (1 - 1 / (4 *
                                                        self.num_params) + 1 / (21 * self.num_params ** 2))

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(self.pop_size, self.num_params)

        print(self.mu)
        print(self.cov)
        print(self.step_size)

        return self.mu + self.step_size * epsilon * np.sqrt(self.cov)

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        # scores preprocess
        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        scores = -np.array(scores)
        idx_sorted = np.argsort(scores)

        # update mean
        old_mu = deepcopy(self.mu)
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        # update evolution paths
        self.p_s = (1 - self.c_s) * self.p_s + \
            np.sqrt(self.c_s * (2 - self.c_s) * self.parents_eff) * \
            (self.mu - old_mu) / self.step_size * 1 / np.sqrt(self.cov)

        tmp_1 = np.linalg.norm(self.p_s) / np.sqrt(self.c_s * (2 - self.c_s)) \
            <= self.chi * (1.4 + 2 / (self.num_params + 1))

        self.p_c = (1 - self.c_c) * self.p_c + \
            tmp_1 * np.sqrt(self.c_c * (2 - self.c_c) * self.parents_eff) * \
            (self.mu - old_mu) / self.step_size

        # update covariance matrix
        tmp_2 = 1 / self.step_size * \
            (solutions[idx_sorted[:self.parents]] - old_mu)

        self.cov = (1 - self.c_cov) * self.cov + \
            self.c_cov * 1 / self.parents_eff * self.p_c * self.p_c + \
            self.c_cov * (1 - 1 / self.parents_eff) * \
            (self.weights @ (tmp_2 * tmp_2))

        # update step size
        self.step_size *= np.exp((self.c_s / self.d_s) *
                                 (np.linalg.norm(self.p_s) / self.chi - 1))

        return idx_sorted[:self.parents]

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and the covariance matrix
        """
        return np.copy(self.mu), np.copy(self.step_size)**2 * np.copy(self.cov)
