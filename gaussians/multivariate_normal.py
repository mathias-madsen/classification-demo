import numpy as np


def squared_mahalanobis_norm(v, M):
    if np.shape(M) == ():
        return np.reshape(M, []) * v ** 2  # stack of scalars
    else:
        return np.sum((v @ M.T) * v, axis=-1)


class MultivariateNormal:

    def __init__(self, mean, cov):
        self.mean = mean
        if np.shape(mean) != ():
            self.dim, = self.mean.shape
            self.cov = cov.reshape([self.dim, self.dim])
            self.precision = np.linalg.inv(self.cov)
            _, self.logabsdet_precision = np.linalg.slogdet(self.precision)
        else:
            self.dim = 1
            self.cov = np.reshape(cov, [])
            self.precision = 1 / self.cov
            self.logabsdet_precision = self.precision

    def logpdf(self, x):
        squares = squared_mahalanobis_norm(x - self.mean, self.precision)
        constant = self.logabsdet_precision - self.dim * np.log(2 * np.pi)
        return -0.5 * (squares - constant)

    # some methods for reasoning about the likely range of log p(x)
    # when x ~ p, using the fact that the Mahalanobis distance from
    # the mean of p is a chi-squared random variable when x ~ p:

    def entropy(self):
        neglogdet = self.dim * np.log(2*np.pi) - self.logabsdet_precision
        return 0.5 * (neglogdet + self.dim)

    def max_logp(self):
        return 0.5*self.logabsdet_precision - 0.5*self.dim*np.log(2 * np.pi)

    def mean_logp(self):
        return self.max_logp() - 0.5*self.dim

    def var_logp(self):
        return self.dim / 2

    def std_logp(self):
        return np.sqrt(self.dim / 2)
