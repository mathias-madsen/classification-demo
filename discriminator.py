import numpy as np

from scipy.stats import multivariate_normal


def biased_moments(vectors):
    """ Conservative estimates of mean vector and covariance matrix. """

    assert not np.any(np.isnan(vectors))
    assert not np.any(np.isinf(vectors))

    length, dim = vectors.shape
    data_weight = length / (length + dim)

    empirical_mean = np.mean(vectors, axis=0)
    biased_mean = data_weight*empirical_mean  # leans towards zero

    devs = vectors - biased_mean
    empirical_cov = devs.T @ devs / len(devs)
    biased_cov = (1 - data_weight)*np.eye(dim) + data_weight*empirical_cov

    assert not np.any(np.isnan(biased_mean))
    assert not np.any(np.isnan(biased_cov))

    return biased_mean, biased_cov


class BiGaussianDiscriminator:

    def __init__(self):
        self.dist_pos = None
        self.dist_neg = None
    
    def fit(self, positive_examples, negative_examples):
        
        print("Fitting . . .")
        self.dist_pos = multivariate_normal(*biased_moments(positive_examples))
        self.dist_neg = multivariate_normal(*biased_moments(negative_examples))
        print("Done fitting.")

    def __call__(self, x):
        if self.dist_pos is None or self.dist_neg is None:
            return np.zeros_like(x[..., 0])
        else:
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