import numpy as np

from gaussians.multivariate_normal import MultivariateNormal
from gaussians.moments_tracker import MomentsTracker
from gaussians.moments_tracker import combine, combine_covariance_only
from gaussians import marginal_log_likelihoods as likes


def pick_best_combination(positive_stats, negative_stats, verbose=False):

    dim, = positive_stats.mean.shape
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

    wpos = positive_stats.count / (positive_stats.count + negative_stats.count)
    wneg = negative_stats.count / (positive_stats.count + negative_stats.count)
    shared_cov = wpos*positive_stats.cov + wneg*negative_stats.cov

    if winner == 0:
        poscov = np.diag(shared_cov.diagonal())
        negcov = np.diag(shared_cov.diagonal())
    elif winner == 1:
        poscov = np.diag(positive_stats.cov.diagonal())
        negcov = np.diag(negative_stats.cov.diagonal())
    elif winner == 2:
        poscov = shared_cov.copy()
        negcov = shared_cov.copy()
    elif winner == 3:
        poscov = positive_stats.cov
        negcov = negative_stats.cov
    else:
        raise Exception("Unpected inddex %s" % (winner,))

    post_pos = combine([
        MomentsTracker(positive_stats.mean, poscov, positive_stats.count),
        MomentsTracker(np.zeros(dim), np.eye(dim), 1.0),
    ])

    post_neg = combine([
        MomentsTracker(negative_stats.mean, negcov, negative_stats.count),
        MomentsTracker(np.zeros(dim), np.eye(dim), 1.0),
    ])

    assert not np.isinf(np.linalg.slogdet(post_pos.cov)[1])
    assert not np.isinf(np.linalg.slogdet(post_neg.cov)[1])

    return post_pos, post_neg


class BiGaussianDiscriminator:

    def __init__(self):
        # the raw empirical stats of the data used to train the class models:
        self.stats_pos = None
        self.stats_neg = None
        # the distributions when taking into account the prior and the relative
        # likelihoods of the different model variants in the light of the data:
        self.dist_pos = None
        self.dist_neg = None
    
    def fit(self, positive_examples, negative_examples, verbose=False):
        self.stats_pos = MomentsTracker.fromdata(positive_examples)
        self.stats_neg = MomentsTracker.fromdata(negative_examples)
        self.fit_with_moments(self.stats_pos, self.stats_neg, verbose=verbose)

    def fit_with_moments(self, positive_stats, negative_stats, verbose=False):
        self.stats_pos = positive_stats
        self.stats_neg = negative_stats
        post_pos, post_neg = pick_best_combination(
            positive_stats,
            negative_stats,
            verbose=verbose,
            )
        self.dist_pos = MultivariateNormal(post_pos.mean, post_pos.cov)
        self.dist_neg = MultivariateNormal(post_neg.mean, post_neg.cov)

    def __call__(self, x):
        if self.dist_pos is None:
            return np.zeros_like(x[..., 0])
        else:
            return self.dist_pos.logpdf(x) - self.dist_neg.logpdf(x)
    
    def save(self, path):
        if self.stats_neg is None or self.stats_pos is None:
            raise ValueError("No stats to save")
        stats_list = [
            self.stats_neg,
            self.stats_pos,
            ]
        means = np.stack([m.mean for m in stats_list], axis=0)
        covs = np.stack([m.cov for m in stats_list], axis=0)
        counts = np.stack([m.count for m in stats_list], axis=0)
        np.savez(path, means=means, covs=covs, counts=counts)

    def load(self, path):
        with np.load(path) as archive:
            means = archive["means"]
            covs = archive["covs"]
            counts = archive["counts"]
        stats_neg = MomentsTracker(means[0], covs[0], counts[0])
        stats_pos = MomentsTracker(means[1], covs[1], counts[1])
        self.fit_with_moments(stats_pos, stats_neg)

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