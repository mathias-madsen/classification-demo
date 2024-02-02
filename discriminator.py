import numpy as np

from scipy.stats import multivariate_normal


class BiGaussianDiscriminator:

    def __init__(self):
        self.dist_pos = None
        self.dist_neg = None
    
    def fit(self, positive_examples, negative_examples):
        
        assert not np.any(np.isnan(positive_examples))
        assert not np.any(np.isinf(positive_examples))
        assert not np.any(np.isnan(negative_examples))
        assert not np.any(np.isinf(negative_examples))

        print("Fitting . . .")

        plength, dim = positive_examples.shape
        pos_data_weight = plength / (plength + dim)
        # assert plength > dim
        pmean = pos_data_weight * np.mean(positive_examples, axis=0)
        pdevs = positive_examples - pmean
        pcov_empirical = pdevs.T @ pdevs / len(pdevs)
        pcov = (1 - pos_data_weight)*np.eye(dim) + pos_data_weight*pcov_empirical
        # pmean = np.mean(positive_examples, axis=0)
        # pcov = np.cov(positive_examples.T, ddof=1)

        assert not np.any(np.isnan(pmean))
        assert not np.any(np.isnan(pcov))
        self.dist_pos = multivariate_normal(pmean, pcov)

        nlength, dim = negative_examples.shape
        neg_data_weight = nlength / (nlength + dim)
        # assert nlength > dim
        nmean = neg_data_weight * np.mean(negative_examples, axis=0)
        ndevs = negative_examples - nmean
        ncov_empirical = ndevs.T @ ndevs / len(ndevs)
        ncov = (1 - neg_data_weight)*np.eye(dim) + neg_data_weight*ncov_empirical
        # nmean = np.mean(negative_examples, axis=0)
        # ncov = np.cov(negative_examples.T, ddof=1)

        assert not np.any(np.isnan(nmean))
        assert not np.any(np.isnan(ncov))
        self.dist_neg = multivariate_normal(nmean, ncov)

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