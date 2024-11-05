import numpy as np


LOG2PI = np.log(2.0 * np.pi)


def squared_mahalanobis_norm(v, M):
    if np.shape(M) == ():
        return np.reshape(M, []) * v ** 2  # stack of scalars
    else:
        return np.sum((v @ M.T) * v, axis=-1)


class MultivariateNormal:

    def __init__(self, mean, cov, count=1.0):
        self.mean = mean
        self.count = count  # so we can instantiate with a MomentsTracker
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
        constant = self.logabsdet_precision - self.dim * LOG2PI
        return -0.5 * (squares - constant)

    def sumlogp(self, mean, cov, count):
        """ Calculate the total log-density of a data set from its stats. """
        constant = self.logabsdet_precision - self.dim * LOG2PI
        meanshift = squared_mahalanobis_norm(mean - self.mean, self.precision)
        covshift = np.trace(self.precision @ cov)
        return 0.5 * count * (constant - meanshift - covshift)

    # some methods for reasoning about the likely range of log p(x)
    # when x ~ p, using the fact that the Mahalanobis distance from
    # the mean of p is a chi-squared random variable when x ~ p:

    def entropy(self):
        neglogdet = self.dim * LOG2PI - self.logabsdet_precision
        return 0.5 * (neglogdet + self.dim)

    def max_logp(self):
        return 0.5 * (self.logabsdet_precision - self.dim*LOG2PI)

    def mean_logp(self):
        return self.max_logp() - 0.5*self.dim

    def var_logp(self):
        return self.dim / 2

    def std_logp(self):
        return np.sqrt(self.dim / 2)
