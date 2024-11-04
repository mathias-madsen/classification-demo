import numpy as np
from scipy import stats
from gaussians.multivariate_normal import MultivariateNormal
from gaussians.multivariate_normal import squared_mahalanobis_norm


def test_multivariate_normal_shapes():

    twodist = MultivariateNormal(np.zeros(2), np.eye(2))
    assert twodist.logpdf(np.ones([5, 2])).shape == (5,)
    assert twodist.logpdf(np.ones([1, 2])).shape == (1,)
    assert twodist.logpdf(np.ones([2])).shape == ()

    onedist = MultivariateNormal(np.zeros(1), np.eye(1))
    assert onedist.logpdf(np.ones([5, 1])).shape == (5,)
    assert onedist.logpdf(np.ones([1, 1])).shape == (1,)
    assert onedist.logpdf(np.ones([1])).shape == ()

    zerodist = MultivariateNormal(0.0, 1.0)
    assert zerodist.logpdf(np.ones([5])).shape == (5,)
    assert zerodist.logpdf(np.ones([])).shape == ()


def test_multivariate_normal_logpdf():

    dim = np.random.randint(1, 15)
    mean = np.random.normal(size=dim)
    cov = stats.wishart.rvs(dim, np.eye(dim))
    num_vectors = np.random.randint(10, 100)
    vectors = np.random.normal(size=(num_vectors, dim))

    logps1 = stats.multivariate_normal.logpdf(vectors, mean, cov)
    logps2 = stats.multivariate_normal(mean, cov).logpdf(vectors)
    logps3 = MultivariateNormal(mean, cov).logpdf(vectors)

    assert logps1.shape == logps2.shape == logps3.shape == (num_vectors,)
    assert np.allclose(logps1, logps2)
    assert np.allclose(logps2, logps3)


def test_multivariate_normal_sumlogp():

    dim = np.random.randint(1, 15)
    mean = np.random.normal(size=dim)
    cov = stats.wishart.rvs(dim, np.eye(dim))
    num_vectors = np.random.randint(10, 100)
    vectors = np.random.normal(size=(num_vectors, dim))

    emp_mean = np.mean(vectors, axis=0)
    emp_cov = np.cov(vectors.T, ddof=0).reshape([dim, dim])  # for 1x1
    emp_count = len(vectors)

    scipy_object = stats.multivariate_normal(mean, cov)
    scipy_sum = scipy_object.logpdf(vectors).sum()

    our_object = MultivariateNormal(mean, cov)
    our_sum = our_object.sumlogp(emp_mean, emp_cov, emp_count)

    assert scipy_sum.shape == our_sum.shape == ()
    assert np.isclose(scipy_sum, our_sum)


def test_multivariate_normal_entropy():

    dim = np.random.randint(1, 15)
    mean = np.random.normal(size=dim)
    cov = stats.wishart.rvs(dim, np.eye(dim))

    entropy1 = stats.multivariate_normal.entropy(mean, cov)
    entropy2 = stats.multivariate_normal(mean, cov).entropy()
    entropy3 = MultivariateNormal(mean, cov).entropy()

    assert np.shape(entropy1) == np.shape(entropy2) == np.shape(entropy3) == ()
    assert np.allclose(entropy1, entropy2)
    assert np.allclose(entropy2, entropy3)


def test_multivariate_normal_logp_statistics():

    dim = np.random.randint(1, 15)
    mean = np.random.normal(size=dim)
    cov = stats.wishart.rvs(dim, np.eye(dim))
    scipy_dist = stats.multivariate_normal(mean, cov)
    logps = scipy_dist.logpdf(scipy_dist.rvs(size=10000))
    custom_dist = MultivariateNormal(mean, cov)
    assert np.all(custom_dist.max_logp() >= logps.max())
    assert np.isclose(custom_dist.mean_logp(), logps.mean(), atol=0.3)
    assert np.isclose(custom_dist.var_logp(), logps.var(), atol=0.3)
    assert np.isclose(custom_dist.std_logp(), logps.std(), atol=0.3)


def test_that_Mahalanobis_is_Euclidean_for_M_equals_I():

    dim = np.random.randint(1, 15)
    num_vectors = np.random.randint(10, 100)
    vectors = np.random.normal(size=(num_vectors, dim))
    squares = squared_mahalanobis_norm(vectors, np.eye(dim))
    assert squares.shape == (num_vectors,)
    assert np.allclose(squares, np.sum(vectors ** 2, axis=-1))


if __name__ == "__main__":

    test_multivariate_normal_shapes()
    test_multivariate_normal_logpdf()
    test_multivariate_normal_sumlogp()
    test_multivariate_normal_entropy()
    test_multivariate_normal_logp_statistics()
    test_that_Mahalanobis_is_Euclidean_for_M_equals_I()