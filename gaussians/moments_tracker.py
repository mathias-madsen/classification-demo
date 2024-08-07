import numpy as np


def combine(moments_objects):
    """ Create a single MomentsTracker objects by mixing several. """
    total = sum(m.count for m in moments_objects)
    mean = np.sum([m.count * m.mean / total for m in moments_objects], axis=0)
    squared_mean = np.outer(mean, mean)
    mean_square = np.sum([m.count / total * (m.cov + np.outer(m.mean, m.mean))
                          for m in moments_objects], axis=0)
    cov = mean_square - squared_mean
    count = sum(m.count for m in moments_objects)
    return MomentsTracker(mean, cov, count)


def mixed_mean(mean, cov, size, mean0, cov0, size0):
    """ Compute the mean of a mixture of two distributions. """
    return (size0 / (size0 + size) * mean +
             size / (size0 + size) * mean0)


def mixed_covariance(mean, cov, size, mean0, cov0, size0):
    """ Compute the covariance of a mixture of two distributions. """
    correction = np.outer(mean - mean0, mean - mean0)
    return (
        size0 / (size0 + size) * cov0 +
        size / (size0 + size) * cov +
        size0 * size / (size0 + size) ** 2 * correction
        )


class MomentsTracker:

    def __init__(self, mean, cov, count=0):
        self.mean = mean
        self.cov = cov
        self.count = count
        self.dim, = np.shape(self.mean)
        self.cov = self.cov.reshape([self.dim, self.dim])  # in case of 1x1
    
    def update_with_single(self, x):
        """ Update the tracker with a single vectorial observation. """
        dim, = np.shape(x)
        assert dim == self.dim
        N0 = self.count
        meandev = np.outer(x - self.mean, x - self.mean)
        self.mean *= N0 / (N0 + 1)
        self.mean += 1 / (N0 + 1) * x
        self.cov *= N0 / (N0 + 1)
        self.cov += N0 / (N0 + 1)**2 * meandev
        self.count += 1

    def update_with_batch(self, x):
        """ Update the tracker with a stack of N observations. """
        N, dim = np.shape(x)
        assert N > 0
        assert dim == self.dim
        ex = np.mean(x, axis=0)
        covx = np.cov(x.T, ddof=0).reshape([dim, dim])
        N0 = self.count
        meandev = np.outer(ex - self.mean, ex - self.mean)
        self.mean *= N0 / (N0 + N)
        self.mean += N / (N0 + N) * ex
        self.cov *= N0 / (N0 + N)
        self.cov += N / (N0 + N) * covx
        self.cov += N0 * N / (N0 + N)**2 * meandev
        self.count += N
    
    def update_with_moments(self, mean, cov, count):
        assert count > 0
        assert mean.shape == (self.dim,)
        meandev = np.outer(mean - self.mean, mean - self.mean)
        self.mean *= self.count / (self.count + count)
        self.mean += count / (self.count + count) * mean
        self.cov *= self.count / (self.count + count)
        self.cov += count / (self.count + count) * cov
        self.cov += self.count * count / (self.count + count)**2 * meandev
        self.count += count

    def __repr__(self):
        return ("MomentsTracker(mean=%r, cov=%r, count=%r)" %
                (self.mean, self.cov, self.count))

    def copy(self):
        """ Make a deep copy of the moments tracker. """
        mean = np.copy(self.mean)
        cov = np.copy(self.cov)
        count = np.copy(self.count)
        return MomentsTracker(mean=mean, cov=cov, count=count)

    @classmethod
    def random(self, dim):
        count = np.random.gamma(1.0)
        mean = np.random.normal(size=dim)
        rootcov = np.random.normal(size=(dim, dim))
        cov = rootcov.T @ rootcov
        return MomentsTracker(mean=mean, cov=cov, count=count)

    @classmethod
    def fromdata(self, sample):
        count = len(sample)
        mean = np.mean(sample, axis=0)
        cov = np.cov(sample.T, ddof=0)
        return MomentsTracker(mean=mean, cov=cov, count=count)
    
    def __iter__(self):
        return iter([self.mean, self.cov, self.count])

    def reset(self):
        """ Set all estimates to null values. """
        self.mean *= 0
        self.cov[:, :] = np.eye(self.cov.shape[1])
        self.count *= 0

