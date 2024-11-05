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
    
    def fit(self, negative_examples, positive_examples, verbose=False):
        self.stats_pos = MomentsTracker.fromdata(positive_examples)
        self.stats_neg = MomentsTracker.fromdata(negative_examples)
        self.fit_with_moments(self.stats_neg, self.stats_pos, verbose=verbose)

    def fit_with_moments(self, negative_stats, positive_stats, verbose=False):
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
        np.savez(path, means=means, covs=covs, counts=counts, version=1)

    def load(self, path):
        with np.load(path) as archive:
            means = archive["means"]
            covs = archive["covs"]
            counts = archive["counts"]
            assert archive["version"] == 1
        stats_neg = MomentsTracker(means[0], covs[0], counts[0])
        stats_pos = MomentsTracker(means[1], covs[1], counts[1])
        self.fit_with_moments(stats_neg, stats_pos)

    @classmethod
    def fromsaved(self, path):
        instance = BiGaussianDiscriminator()
        instance.load(path)
        return instance
