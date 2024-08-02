import numpy as np
from scipy import stats


def squared_mahalanobis(vectors, matrix, axis=-1):
    """ Squared Mahalanobis norm of a vector wrt a matrix. """
    return np.sum(vectors * (vectors @ matrix.T), axis=axis)


def log_affinity(mean1, precision1, mean2, precision2):
    """ Bhattacharyya coefficient between two multivariate Gaussians.
    
    The Bhattacharyya coefficient is the integral of `sqrt(p * q)` over
    the entire sample space and can be used to lower-bound the smallest
    achievable error in a binary hypothesis test.
    """
    ave_precision = (precision1 + precision2) / 2
    ave_mean = (precision1 @ mean1 + precision2 @ mean2) / 2
    return (
        + 0.5 * squared_mahalanobis(ave_mean, np.linalg.inv(ave_precision))
        - 0.25 * squared_mahalanobis(mean1, precision1)
        - 0.25 * squared_mahalanobis(mean2, precision2)
        - 0.5 * np.linalg.slogdet(ave_precision).logabsdet
        + 0.25 * np.linalg.slogdet(precision1).logabsdet
        + 0.25 * np.linalg.slogdet(precision2).logabsdet
        )


def lower_bound_on_error(mean1, cov1, mean2, cov2):
    """ Compute a lower bound on best possible error rate. """
    logb = log_affinity(mean1, cov1, mean2, cov2)
    return 0.5 * (1 - np.sqrt(1 - np.exp(2 * logb)))


def numerically_integrate(function, bounds, pts_per_dim):
    """ Approximate the integral of f over a D-dimensional rectangle. """
    spans = [np.linspace(a, b, pts_per_dim) for a, b in bounds]
    volume_element = np.prod([np.mean(np.diff(s)) for s in spans])
    spans = np.meshgrid(*spans)
    sample_points = np.stack([s.flatten() for s in spans], axis=1)
    return np.sum(function(sample_points)) * volume_element


if __name__ == "__main__":

    dim = 2
    Q1, Q2 = stats.wishart.rvs(dim, np.eye(dim), size=2)
    mu1, mu2 = np.random.normal(size=(2, dim))

    dist1 = stats.multivariate_normal(mean=mu1, cov=np.linalg.inv(Q1))
    dist2 = stats.multivariate_normal(mean=mu2, cov=np.linalg.inv(Q2))

    points = np.random.normal(size=(1000, dim))

    logp1 = 0.5 * (
        np.linalg.slogdet(1 / (2 * np.pi) * Q1).logabsdet
        - squared_mahalanobis(mu1 - points, Q1)
        )
    assert np.allclose(logp1, dist1.logpdf(points))

    logp2 = 0.5 * (
        np.linalg.slogdet(1 / (2 * np.pi) * Q2).logabsdet
        - squared_mahalanobis(mu2 - points, Q2)
        )
    assert np.allclose(logp2, dist2.logpdf(points))

    logsqrtpq = 0.5 * (dist1.logpdf(points) + dist2.logpdf(points))
    sqrtpq = np.sqrt(dist1.pdf(points) * dist2.pdf(points))
    assert np.allclose(sqrtpq, np.exp(logsqrtpq))

    similarity = numerically_integrate(
        function=lambda t: np.sqrt(dist1.pdf(t) * dist2.pdf(t)),
        bounds=[(-10, +10) for _ in range(dim)],
        pts_per_dim=1000
        )

    print(np.exp(log_affinity(mu1, Q1, mu2, Q2)))
    print(similarity)
    print()

    error_rate_1 = numerically_integrate(
        function=lambda t: dist1.pdf(t) * (dist1.pdf(t) < dist2.pdf(t)),
        bounds=[(-10, +10) for _ in range(dim)],
        pts_per_dim=100
        )
    sample1 = dist1.rvs(size=10000)
    average_error_1 = np.mean(dist1.logpdf(sample1) < dist2.logpdf(sample1))

    error_rate_2 = numerically_integrate(
        function=lambda t: dist2.pdf(t) * (dist1.pdf(t) > dist2.pdf(t)),
        bounds=[(-10, +10) for _ in range(dim)],
        pts_per_dim=1000
        )
    sample2 = dist2.rvs(size=10000)
    average_error_2 = np.mean(dist2.logpdf(sample2) < dist1.logpdf(sample2))

    print("Error rate 1: %.3f ~= %.3f" % (error_rate_1, average_error_1))
    print("Error rate 2: %.3f ~= %.3f" % (error_rate_2, average_error_2))
    print()

    error_rate = (error_rate_1 + error_rate_2) / 2
    average_error = (average_error_1 + average_error_2) / 2
    error_bound = lower_bound_on_error(mu1, Q1, mu2, Q2)

    print("Error: %.3f <= %.3f ~= %.3f" % (error_bound, error_rate, average_error))
    print()
