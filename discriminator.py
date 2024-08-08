import numpy as np
from scipy.stats import multivariate_normal

from gaussians.moments_tracker import MomentsTracker, combine


# def biased_moments(vectors, prior_mean=None, prior_cov=None):
#     """ Conservative estimates of mean vector and covariance matrix. """

#     assert not np.any(np.isnan(vectors))
#     assert not np.any(np.isinf(vectors))

#     length, dim = vectors.shape
#     data_weight = length / (length + dim)

#     if prior_mean is None:
#         prior_mean = np.zeros(dim)
    
#     if prior_cov is None:
#         prior_cov = np.eye(dim)

#     empirical_mean = np.mean(vectors, axis=0)
#     biased_mean = (1 - data_weight)*prior_mean + data_weight*empirical_mean

#     devs = vectors - biased_mean
#     empirical_cov = devs.T @ devs / len(devs)
#     biased_cov = (1 - data_weight)*prior_cov + data_weight*empirical_cov

#     assert not np.any(np.isnan(biased_mean))
#     assert not np.any(np.isnan(biased_cov))

#     return biased_mean, biased_cov


class BiGaussianDiscriminator:

    def __init__(self, dim=None):
        if dim is not None:
            self.dist_pos = multivariate_normal(np.zeros(dim), np.eye(dim))
            self.dist_neg = multivariate_normal(np.zeros(dim), np.eye(dim))
        else:
            self.dist_pos = None
            self.dist_neg = None
    
    def fit(self, positive_examples, negative_examples):
        
        print("Fitting . . .")
        positive_stats = MomentsTracker.fromdata(positive_examples)
        negative_stats = MomentsTracker.fromdata(negative_examples)
        self.fit_with_moments(positive_stats, negative_stats)
        del positive_stats
        del negative_stats
        print("Done fitting.")

    def fit_with_moments(self, positive_stats, negative_stats):

        xbar = combine([positive_stats, negative_stats])
        dim, = xbar.mean.shape
        N = xbar.count  # N1 + N2
        mean = xbar.mean
        cov = xbar.cov + (dim / (dim + N))**2 * np.outer(mean, mean)
        poolmean = N / (N + dim) * mean
        poolcov = N / (N + dim) * cov + dim / (N + dim) * np.eye(dim)

        N1 = positive_stats.count
        N2 = negative_stats.count
        mean1 = N1 / (N1 + dim) * positive_stats.mean + dim / (N1 + dim) * poolmean
        mean2 = N2 / (N2 + dim) * negative_stats.mean + dim / (N2 + dim) * poolmean
        del xbar

        if self.dist_pos is not None:
            self.dist_pos.mean[:] = mean1
            self.dist_pos.cov[:] = poolcov
            self.dist_neg.mean[:] = mean2
            self.dist_neg.cov[:] = poolcov
        else:
            self.dist_pos = multivariate_normal(mean1, poolcov)
            self.dist_neg = multivariate_normal(mean2, poolcov)

    def __call__(self, x):
        if self.dist_pos is None:
            return np.zeros_like(x[..., 0])
        else:
            return self.dist_pos.logpdf(x) - self.dist_neg.logpdf(x)


def test_bigaussian_discriminator():

    dim = 5
    size1 = 17
    size2 = 13
    x1 = np.linspace(-3.0, +2.0, size1 * dim).reshape([size1, dim])
    x2 = np.linspace(-2.0, +3.0, size2 * dim).reshape([size2, dim])

    mean1 = np.array(
        [-0.51767355, -0.4585121 , -0.39935065, -0.3401892 , -0.28102775]
        )

    mean2 = np.array(
        [0.20020786, 0.27272298, 0.3452381 , 0.41775321, 0.49026833]
        )
    
    cov1 = np.array([
        [2.16468973, 2.02536342, 2.02889426, 2.0324251 , 2.03595594],
        [2.02536342, 2.17190413, 2.03273055, 2.03641412, 2.04009769],
        [2.02889426, 2.03273055, 2.17942399, 2.04040314, 2.04423943],
        [2.0324251 , 2.03641412, 2.04040314, 2.1872493 , 2.04838118],
        [2.03595594, 2.04009769, 2.04423943, 2.04838118, 2.19538006],
        ])
    cov2 = np.array([
        [2.16468973, 2.02536342, 2.02889426, 2.0324251 , 2.03595594],
        [2.02536342, 2.17190413, 2.03273055, 2.03641412, 2.04009769],
        [2.02889426, 2.03273055, 2.17942399, 2.04040314, 2.04423943],
        [2.0324251 , 2.03641412, 2.04040314, 2.1872493 , 2.04838118],
        [2.03595594, 2.04009769, 2.04423943, 2.04838118, 2.19538006],
        ])

    assert np.linalg.det(cov1) > 1e-5
    assert np.linalg.det(cov2) > 1e-5

    discriminator = BiGaussianDiscriminator()
    discriminator.fit(x1, x2)

    assert np.allclose(discriminator.dist_pos.mean, mean1)
    assert np.allclose(discriminator.dist_neg.mean, mean2)
    assert np.allclose(discriminator.dist_pos.cov, cov1)
    assert np.allclose(discriminator.dist_neg.cov, cov2)


if __name__ == "__main__":

    test_bigaussian_discriminator()

    from gaussians.moments_tracker import MomentsTracker, combine

    dim = 5
    size1, size2 = np.random.randint(10, 100, size=2)
    loc1, loc2 = np.random.normal(size=(2, dim), scale=5.0)
    scale1, scale2 = np.random.normal(size=(2, dim, dim))
    cov1 = scale1 @ scale1.T
    cov2 = scale2 @ scale2.T

    x1 = np.random.multivariate_normal(loc1, cov1, size=size1)
    x2 = np.random.multivariate_normal(loc2, cov2, size=size2)

    np.set_printoptions(precision=5, linewidth=120, suppress=True)

    posex = MomentsTracker.fromdata(x1)
    negex = MomentsTracker.fromdata(x2)

    moments0 = MomentsTracker(np.zeros(dim), np.eye(dim), count=dim)
    pooled_moments = combine([moments0, posex, negex])
    pooled_moments.count = dim
    m1 = combine([pooled_moments, posex])
    m2 = combine([pooled_moments, negex])

    xbar1 = np.mean(x1, axis=0)
    xbar2 = np.mean(x2, axis=0)
    xbar = size1 / (size1 + size2) * xbar1 + size2 / (size1 + size2) * xbar2
    mean = (size1 + size2) / (size1 + size2 + dim) * xbar
    mean1 = size1 / (size1 + dim) * xbar1 + dim / (size1 + dim) * mean
    mean2 = size2 / (size2 + dim) * xbar2 + dim / (size2 + dim) * mean

    discriminator = BiGaussianDiscriminator()
    discriminator.fit(x1, x2)

    # print(loc1)
    print(discriminator.dist_pos.mean)
    print(mean1)
    print(m1.mean)
    print()
    # print(loc2)
    print(discriminator.dist_neg.mean)
    print(mean2)
    print(m2.mean)
    print()
    # print(discriminator(x1).mean())
    # print(discriminator(x2).mean())
    # print()