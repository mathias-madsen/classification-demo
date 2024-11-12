import numpy as np

from gaussians.multivariate_normal import MultivariateNormal


class BiGaussianDiscriminator:

    def __init__(self):
        self.dist_pos = None
        self.dist_neg = None
        self.n_stds = 30.0  # for assigning prob to 'neither'
    
    def set_stats(self, neg_stats, pos_stats):
        self.dist_pos = MultivariateNormal(*pos_stats)
        self.dist_neg = MultivariateNormal(*neg_stats)

    def smallest_expected_logprob(self):
        pos_bound = (self.dist_pos.mean_logp() -
                     self.n_stds * self.dist_pos.std_logp())
        neg_bound = (self.dist_neg.mean_logp() -
                     self.n_stds * self.dist_neg.std_logp())
        return min(pos_bound, neg_bound)

    def __call__(self, x):
        pos = self.dist_pos.logpdf(x)
        neg = self.dist_neg.logpdf(x)
        neither_logprob = self.smallest_expected_logprob()
        neither = neither_logprob * np.ones_like(pos)
        logprobs = np.array([neg, pos, neither])
        logprobs -= np.max(logprobs)
        logsumexp = np.log(np.sum(np.exp(logprobs)))
        return logprobs - logsumexp
    
    def evaluate(self, test_codes, labels):
        return np.argmax(self(test_codes), axis=0) == labels
    
    def classify(self, vectors):
        """ Return a class index (0 or 1), or 2 for neither. """
        return np.argmax(self(vectors), axis=0)

    def save(self, path):
        if self.dist_pos is None or self.dist_neg is None:
            raise ValueError("No parameters to save yet")
        dist_list = [
            self.dist_neg,
            self.dist_pos,
            ]
        means = np.stack([m.mean for m in dist_list], axis=0)
        covs = np.stack([m.cov for m in dist_list], axis=0)
        counts = np.stack([m.count for m in dist_list], axis=0)
        np.savez(path, means=means, covs=covs, counts=counts)

    def load(self, path):
        with np.load(path) as archive:
            means = archive["means"]
            covs = archive["covs"]
            counts = archive.get("counts", (1.0, 1.0))
        stats_neg = means[0], covs[0], counts[0]
        stats_pos = means[1], covs[1], counts[1]
        self.set_stats(stats_neg, stats_pos)

    @classmethod
    def fromsaved(cls, path):
        instance = cls()
        instance.load(path)
        return instance