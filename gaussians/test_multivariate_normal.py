import numpy as np
from scipy import stats
from gaussians.multivariate_normal import MultivariateNormal
from gaussians.multivariate_normal import squared_mahalanobis_norm


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



def test_that_Mahalanobis_is_Euclidean_for_M_equals_I():

    dim = np.random.randint(1, 15)
    num_vectors = np.random.randint(10, 100)
    vectors = np.random.normal(size=(num_vectors, dim))
    squares = squared_mahalanobis_norm(vectors, np.eye(dim))
    assert squares.shape == (num_vectors,)
    assert np.allclose(squares, np.sum(vectors ** 2, axis=-1))


if __name__ == "__main__":

    test_multivariate_normal_logpdf()
    test_multivariate_normal_entropy()
    test_that_Mahalanobis_is_Euclidean_for_M_equals_I()