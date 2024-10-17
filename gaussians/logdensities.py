import numpy as np
from scipy.stats import multivariate_normal, wishart, chi2


def logp_ppf(cov, p):
    """ Quantile function for log p(X) for multivariate normal X ~ p.
    
    Parameters:
    -----------
    cov : positive-definite matrix of shape [D, D]
        Covariance matrix of the distribution.
    p : float in [0, 1]
        The left-tail probability whose upper bound we want to compute.
    
    Returns:
    --------
    t : float
        A threshold such that `P(log p(X) < t) = p`.
    """
    dim, _ = cov.shape
    threshold_mahalanobis = chi2.ppf(1 - p, df=dim)
    _, logabsdet = np.linalg.slogdet(2 * np.pi * cov)
    return -0.5 * (threshold_mahalanobis + logabsdet)


def test_logp_ppf():

    dim = np.random.randint(10, 30)
    cov = wishart.rvs(dim, np.eye(dim))
    mean = np.zeros(dim)
    dist = multivariate_normal(mean, cov)
    logps = dist.logpdf(dist.rvs(size=100000))
    
    for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predicted = logp_ppf(cov, fraction)
        estimated = np.percentile(logps, 100 * fraction)
        assert np.isclose(predicted, estimated, atol=0.1)


if __name__ == "__main__":

    from scipy.stats import kstest
    from matplotlib import pyplot as plt

    dim = np.random.randint(10, 30)
    cov = wishart.rvs(dim, np.eye(dim))
    mean = np.zeros(dim)
    dist = multivariate_normal(mean, cov)
    logps = dist.logpdf(dist.rvs(size=1000))
    _, logabsdet = np.linalg.slogdet(2 * np.pi * cov)

    # The Mahalanobis distances from the mean are a
    # linear function of the logarithmic densities:
    mahalanobis = -2*logps - logabsdet

    # The Mahalanobis distances are chi^2-distributed:
    cdf = lambda s: chi2.cdf(s, df=dim)
    test_result = kstest(mahalanobis, cdf)
    assert test_result.statistic < 0.1
    assert test_result.pvalue > 1e-6

    span = np.linspace(0, 3 * dim, 1000)
    plt.figure(figsize=(12, 5))
    plt.hist(mahalanobis, bins=30, density=True, alpha=0.5)
    plt.plot(span, chi2.pdf(span, df=dim), lw=3, alpha=0.5)
    plt.ylabel("Probability density")
    plt.tight_layout()
    plt.show()

    N = len(mahalanobis)
    x = np.sort(mahalanobis)
    y = np.arange(N) / (N - 1)
    span = np.linspace(0, 3 * dim, 1000)
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, ".", alpha=0.5)
    plt.plot(span, chi2.cdf(span, df=dim), lw=3, alpha=0.5)
    plt.ylabel("Cumulative probability")
    plt.tight_layout()
    plt.show()