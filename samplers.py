import numpy as np
import operator

from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.misc import logsumexp


class BasicSampler():

    """
    Simple sampler that relies on the ask method
    of the optimizers
    """

    def __init__(self, sample_archive, thetas_archive, **kwargs):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        return

    def ask(self, pop_size, optimizer):
        return optimizer.ask(pop_size), 0, 0, 0, 0, 0


class IMSampler():

    """
    Importance mixing sampler optimized for diagonal covariance matrix
    """

    def __init__(self, sample_archive, thetas_archive, **kwargs):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        self.k = kwargs["k"]

    def ask(self, pop_size, optimizer):

        if len(self.sample_archive) < pop_size:
            return optimizer.ask(pop_size), 0, [], []

        # misc
        n_reused = 0
        n_sampled = 0
        theta = self.thetas_archive[-1]
        mu = theta.mu
        cov = theta.cov

        scores_reused = []
        idx_reused = []
        params = np.zeros((pop_size, mu.shape[0]))

        # current pdf
        def new_log_pdf(z):
            return norm.logpdf(z, loc=mu, scale=np.sqrt(cov)).sum()

        # iterating over k last populations
        for j in range(min(self.k, len(self.thetas_archive) - 1)):

            # old distribution parameters
            id_theta = len(self.thetas_archive) - 2 - j
            old_theta = self.thetas_archive[id_theta]

            # iterating over population
            for i in range(len(old_theta.samples)):

                old_mu = old_theta.mu
                old_cov = old_theta.cov

                # old individual
                id_sample = old_theta.samples[i]
                sample = self.sample_archive[id_sample]

                # old pdf
                def old_log_pdf(z):
                    return norm.logpdf(z, loc=old_mu, scale=np.sqrt(old_cov)).sum()

                if n_reused + n_sampled < pop_size:

                    param = sample.params
                    u = np.random.uniform(0, 1)

                    # rejection sampling
                    if np.log(u) < new_log_pdf(param) - old_log_pdf(param):

                        params[n_reused] = param
                        scores_reused.append(sample.score)
                        idx_reused.append(
                            len(self.sample_archive) - pop_size + i)
                        n_reused += 1

                if n_reused + n_sampled < pop_size:

                    param = optimizer.ask(1).reshape(-1)
                    u = np.random.uniform(0, 1)

                    # rejection sampling
                    if np.log(1 - u) >= old_log_pdf(param) - new_log_pdf(param):
                        params[-n_sampled-1] = param
                        n_sampled += 1

            if n_reused + n_sampled >= pop_size:
                break

        # filling the rest of the population
        cpt = n_reused + n_sampled
        while cpt < pop_size:

            param = optimizer.ask(1).reshape(-1)
            params[cpt - n_sampled] = param
            cpt += 1

        return params, n_reused, idx_reused, scores_reused
