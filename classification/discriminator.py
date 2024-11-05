import numpy as np

from gaussians.multivariate_normal import MultivariateNormal
from gaussians.moments_tracker import MomentsTracker
from gaussians.marginal_log_likelihoods import pick_best_combination


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
    
    def evaluate(self, vectors, labels):
        case_0 = (self(vectors) < -1e-5) * (labels == 0)
        case_1 = (self(vectors) > +1e-5) * (labels == 1)
        return case_0 + case_1
    
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
