import numpy as np

from Optimizers import Adam


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
        grad = -1/(self.sigma * self.pop_size) * np.dot(reward, epsilon)

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step

    def add(self, grad, fitness):
        """
        Adds new "gradient" to U
        """
        grad = grad / np.linalg.norm(grad)
        self.U[:, -1] = grad

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)
