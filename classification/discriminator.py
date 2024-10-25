import numpy as np

from gaussians.multivariate_normal import MultivariateNormal


class BiGaussianDiscriminator:

    def __init__(self):
        self.dist_pos = None
        self.dist_neg = None
    
    def set_stats(self, neg_stats, pos_stats):
        self.dist_pos = MultivariateNormal(*pos_stats)
        self.dist_neg = MultivariateNormal(*neg_stats)

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
    def fromsaved(self, path):
        instance = BiGaussianDiscriminator()
        instance.load(path)
        return instance

    def evaluate(self, codes, labels):
        """ Return booleans indicating where the model is correct. """
        logits = self(codes)
        return (
            (logits > 1e-5) * (labels == 1) +
            (logits < -1e-5) * (labels == 0)
            )
