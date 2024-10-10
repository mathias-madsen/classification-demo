import numpy as np
from scipy.stats import multivariate_normal

from gaussians.moments_tracker import MomentsTracker, combine
from gaussians import marginal_log_likelihoods as likes


class BiGaussianDiscriminator:

    def __init__(self, dim=None):
        if dim is not None:
            self.dist_pos = multivariate_normal(np.zeros(dim), np.eye(dim))
            self.dist_neg = multivariate_normal(np.zeros(dim), np.eye(dim))
        else:
            self.dist_pos = None
            self.dist_neg = None
    
    def fit(self, positive_examples, negative_examples, verbose=False):
        
        positive_stats = MomentsTracker.fromdata(positive_examples)
        negative_stats = MomentsTracker.fromdata(negative_examples)
        self.fit_with_moments(positive_stats, negative_stats, verbose=verbose)
        del positive_stats
        del negative_stats

    def fit_with_moments(self, positive_stats, negative_stats, verbose=False):

        dim, = positive_stats.mean.shape

        # prior_stats = MomentsTracker(np.zeros(dim), np.eye(dim), dim)
        prior_stats = MomentsTracker(np.zeros(dim), np.eye(dim), 1.0)

        sepvar = likes.marginal_logp_separate_variances(
            positive_stats,
            negative_stats,
            prior_stats,
            prior_stats,
            1.0,
        )

        sharevar = likes.marginal_logp_separate_variances(
            positive_stats,
            negative_stats,
            prior_stats,
            prior_stats,
            1.0,
        )

        sepcov = likes.marginal_logp_separate_covariances(
            positive_stats,
            negative_stats,
            prior_stats,
            prior_stats,
            1.0,
        )

        sharecov = likes.marginal_logp_shared_covariance(
            positive_stats,
            negative_stats,
            prior_stats,
            prior_stats,
            1.0,
        )

        winner = np.argmax([sharevar, sepvar, sharecov, sepcov])

        if verbose:

            print("Marginal log-likelihoods:")
            print("-------------------------")
            print("shared diagonal matrix:   ", sharevar)
            print("two diagonal matrices:    ", sepvar)
            print("shared covariance matrix: ", sharecov)
            print("two covariance matrices:  ", sepcov)
            print()

            if winner == 0:
                print("Best model: a shared diagonal matrix.\n")
            elif winner == 1:
                print("Best model: two separate diagonal matrices.\n")
            elif winner == 2:
                print("Best model: a shared covariance matrix.\n")
            elif winner == 3:
                print("Best model: two separate covariance matrices.\n")
            else:
                raise Exception("Unpected inddex %s" % (winner,))

        combined_stats = combine([positive_stats, negative_stats])
        posterior_shared = combine([combined_stats, prior_stats])
        posterior_pos = combine([positive_stats, prior_stats])
        posterior_neg = combine([negative_stats, prior_stats])
        posmean = posterior_pos.mean
        negmean = posterior_neg.mean

        if winner == 0:
            poscov = np.diag(posterior_shared.cov.diagonal())
            negcov = np.diag(posterior_shared.cov.diagonal())
        elif winner == 1:
            poscov = np.diag(posterior_pos.cov.diagonal())
            negcov = np.diag(posterior_neg.cov.diagonal())
        elif winner == 2:
            poscov = posterior_shared.cov.copy()
            negcov = posterior_shared.cov.copy()
        elif winner == 3:
            poscov = posterior_pos.cov
            negcov = posterior_neg.cov
        else:
            raise Exception("Unpected inddex %s" % (winner,))

        if self.dist_pos is not None:
            self.dist_pos.mean[:] = posmean
            self.dist_pos.cov[:] = poscov
            self.dist_neg.mean[:] = negmean
            self.dist_neg.cov[:] = negcov
        else:
            self.dist_pos = multivariate_normal(posmean, poscov)
            self.dist_neg = multivariate_normal(negmean, negcov)

    def __call__(self, x):
        if self.dist_pos is None:
            return np.zeros_like(x[..., 0])
        else:
            return self.dist_pos.logpdf(x) - self.dist_neg.logpdf(x)
    
    def save(self, path):
        if self.dist_neg is None or self.dist_pos is None:
            raise ValueError("No parameters to save")
        dists = [self.dist_neg, self.dist_pos]  # CANONICAL ORDER
        cluster_means = np.stack([d.mean for d in dists], axis=0)
        cluster_covs = np.stack([d.cov for d in dists], axis=0)
        np.savez(path, cluster_means=cluster_means, cluster_covs=cluster_covs)

    def load(self, path):
        with np.load(path) as archive:
            cluster_means = archive["cluster_means"]
            cluster_covs = archive["cluster_covs"]
        # NEGATIVE is number 1 (1-based):
        self.dist_neg = multivariate_normal(cluster_means[0], cluster_covs[0])
        # POSITIVE is number 2 (1-based):
        self.dist_pos = multivariate_normal(cluster_means[1], cluster_covs[1])

    @classmethod
    def fromsaved(self, path):
        instance = BiGaussianDiscriminator()
        instance.load(path)
        return instance


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