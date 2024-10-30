import numpy as np


def squared_mahalanobis_norm(v, M):
    return np.sum((v @ M.T) * v, axis=-1)


def logpdf(x, mean, precision, logabsdet_precision):
    dim, = mean.shape
    squares = squared_mahalanobis_norm(x - mean, precision)
    constant = logabsdet_precision - dim * np.log(2 * np.pi)
    return -0.5 * (squares - constant)


class MultivariateNormal:

    def __init__(self, mean, cov):
        self.mean = mean
        self.precision = np.linalg.inv(cov)
        _, self.logabsdet_precision = np.linalg.slogdet(self.precision)

    def logpdf(self, x):
        return logpdf(x, self.mean, self.precision, self.logabsdet_precision)

    def entropy(self):
        dim, = self.mean.shape  # will fail for scalar mean and cov
        return 0.5 * (dim + dim*np.log(2*np.pi) - self.logabsdet_precision)


