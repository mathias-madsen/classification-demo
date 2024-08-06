import numpy as np

from scipy.stats import multivariate_normal


def biased_moments(vectors, prior_mean=None, prior_cov=None):
    """ Conservative estimates of mean vector and covariance matrix. """

    assert not np.any(np.isnan(vectors))
    assert not np.any(np.isinf(vectors))

    length, dim = vectors.shape
    data_weight = length / (length + dim)

    if prior_mean is None:
        prior_mean = np.zeros(dim)
    
    if prior_cov is None:
        prior_cov = np.eye(dim)

    empirical_mean = np.mean(vectors, axis=0)
    biased_mean = (1 - data_weight)*prior_mean + data_weight*empirical_mean

    devs = vectors - biased_mean
    empirical_cov = devs.T @ devs / len(devs)
    biased_cov = (1 - data_weight)*prior_cov + data_weight*empirical_cov

    assert not np.any(np.isnan(biased_mean))
    assert not np.any(np.isnan(biased_cov))

    return biased_mean, biased_cov


class BiGaussianDiscriminator:

    def __init__(self, dim=205):
        self.dist_pos = multivariate_normal(np.zeros(dim), np.eye(dim))
        self.dist_neg = multivariate_normal(np.zeros(dim), np.eye(dim))
    
    def fit(self, positive_examples, negative_examples):
        
        print("Fitting . . .")
        pooled = np.concatenate([positive_examples, negative_examples], axis=0)
        mean, cov = biased_moments(pooled)
        pos_mean, pos_cov = biased_moments(positive_examples, mean, cov)
        # self.dist_pos = multivariate_normal(pos_mean, pos_cov)
        self.dist_pos = multivariate_normal(pos_mean, cov)
        neg_mean, neg_cov = biased_moments(negative_examples, mean, cov)
        # self.dist_neg = multivariate_normal(neg_mean, neg_cov)
        self.dist_neg = multivariate_normal(neg_mean, cov)
        print("Done fitting.")

    def __call__(self, x):
        return self.dist_pos.logpdf(x) - self.dist_neg.logpdf(x)



if __name__ == "__main__":

    loc1, loc2 = np.random.normal(size=(2, 512), scale=5.0)
    scale1, scale2 = np.random.normal(size=(2, 512, 512))
    cov1 = scale1 @ scale1.T
    cov2 = scale2 @ scale2.T

    x1 = np.random.multivariate_normal(loc1, cov1, size=1200)
    x2 = np.random.multivariate_normal(loc1, cov1, size=1300)

    discriminator = BiGaussianDiscriminator()
    discriminator.fit(x1, x2)

    print(discriminator(x1).shape)
    print(discriminator(x1).mean())
    print()
    print(discriminator(x2).shape)
    print(discriminator(x2).mean())
    print()