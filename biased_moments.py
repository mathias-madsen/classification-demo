import numpy as np
# from scipy import stats


def logsumexp(x, axis=None, keepdims=False):
    """ Compute log(sum(exp(x))) in a numerically stable way. """
    lognorm = np.max(x, axis=axis, keepdims=True)
    summed = np.sum(np.exp(x - lognorm), axis=axis, keepdims=keepdims)
    result = np.log(summed) + lognorm
    return np.reshape(result, summed.shape)


class BiasedMoments:
    """ A mean and covariance that can be updated as data comes in. """

    def __init__(self, mean, cov, df):
        self.mean = mean
        self.cov = cov
        self.df = df

    def update_with_single_observation(self, x):
        squared_mean_change = np.outer(self.mean - x, self.mean - x)
        self.cov *= self.df/(self.df + 1)
        self.cov += self.df/(self.df + 1)**2 * squared_mean_change
        self.mean = self.df/(self.df + 1)*self.mean + 1/(self.df + 1)*x
        self.df += 1


class MixedGaussianModel(BiasedMoments):
    """ A mixture of a diagonal and a full Gaussian model. """

    def __init__(self, *args, **kwargs):
        self.models = ["full", "diagonal", "shared"]
        self.log_weights = np.log([1/3, 1/3, 1/3])
        super().__init__(*args, **kwargs)

    def compute_full_log_like(self, x):
        return stats.multivariate_t.logpdf(
            x,
            df=self.df,
            loc=self.mean,
            shape=self.cov,  # SHAPE (cov) not scale
            )

    def compute_diag_log_like(self, x):
        scales = self.cov.diagonal() ** 0.5,  # SCALE (stds) not shape
        dimlogs = stats.t.logpdf(x, df=self.df, loc=self.mean, scale=scales)
        return dimlogs.sum(axis=-1)  # sum over DIM axis
    
    def compute_diag_log_like(self, x):
        scale = self.cov.diagonal().mean() ** 0.5,  # single scale
        dimlogs = stats.t.logpdf(x, df=self.df, loc=self.mean, scale=scale)
        return dimlogs.sum(axis=-1)  # sum over DIM axis

    def compute_shared_log_like(self, x):
        return stats.t.logpdf(
            x,
            df=self.df,
            loc=self.mean,
            scale=self.cov.diagonal() ** 0.5,  # SCALE (stds) not shape
            ).sum(axis=-1)  # sum over DIM axis

    def compute_fixed_log_like(self, x):
        return stats.norm.logpdf(x, loc=self.mean, scale=1.0).sum(axis=-1)

    def compute_all_log_likes(self, x):
        return (
            self.compute_full_log_like(x),
            self.compute_diag_log_like(x),
            self.compute_shared_log_like(x),
            # self.compute_fixed_log_like(x),
        )

    def compute_mixed_log_like(self, x):
        loglikes = self.compute_all_log_likes(x)
        return logsumexp(self.log_weights + loglikes, axis=0)

    def update_with_single_observation(self, x):
        self.log_weights += self.compute_all_log_likes(x)
        self.log_weights -= logsumexp(self.log_weights, axis=0)
        super().update_with_single_observation(x)


def _test_biased_moments_with_df_set_to_zero():

    nobs = 20
    dim = 3
    observations = np.random.normal(size=(nobs, dim))
    moments = BiasedMoments(np.zeros(dim), np.eye(dim), df=0)
    for x in observations:
        moments.update_with_single_observation(x)
    direct_mean = np.mean(observations, axis=0)
    direct_cov = np.cov(observations.T, ddof=0)
    assert np.allclose(direct_mean, moments.mean)
    assert np.allclose(direct_cov, moments.cov)


def _test_biased_moments_mean_with_df_equal_to_data_set_size():

    nobs = 20
    dim = np.random.randint(1, 10)
    observations = np.random.normal(size=(nobs, dim))
    moments = BiasedMoments(np.zeros(dim), np.eye(dim), df=nobs)
    for x in observations:
        moments.update_with_single_observation(x)
    direct_mean = np.mean(observations, axis=0)
    assert np.allclose(0.5*direct_mean, moments.mean)


def _test_biased_moments_cov_with_df_equal_to_data_set_size():

    nobs = 20
    dim = np.random.randint(1, 10)
    observations = np.random.normal(size=(nobs, dim))
    observations -= np.mean(observations, axis=0)
    moments = BiasedMoments(np.zeros(dim), np.eye(dim), df=nobs)
    for x in observations:
        moments.update_with_single_observation(x)
    direct_cov = np.cov(observations.T, ddof=0)
    assert np.allclose(0.5*direct_cov + 0.5*np.eye(dim), moments.cov)


def _test_biased_moments_with_very_large_data_set_size():

    nobs = 10000
    dim = 3
    true_mean = np.random.normal(size=dim)
    scale = np.random.normal(size=(dim, dim))
    true_cov = scale @ scale.T
    observations = stats.multivariate_normal.rvs(
        mean=true_mean,
        cov=true_cov,
        size=nobs,
        )
    moments = BiasedMoments(np.zeros(dim), np.eye(dim), df=0.5)
    for x in observations:
        moments.update_with_single_observation(x)
    direct_mean = np.mean(observations, axis=0)
    direct_cov = np.cov(observations.T, ddof=0)
    # because the data set size is large, the bias is almost gone:
    assert np.allclose(direct_mean, moments.mean, atol=0.01)
    assert np.allclose(direct_cov, moments.cov, atol=0.01)
    # but this is not exact:
    assert not np.all(direct_mean == moments.mean)
    assert not np.all(direct_cov == moments.cov)


def _test_that_logsumexp_agrees_with_naive_implementation_in_simple_cases():

    vector = [3., 5.]
    indirect = logsumexp(vector)
    direct = np.log(np.sum(np.exp(vector)))
    assert np.isclose(indirect, direct)

    vector = [-2., 5., -7]
    indirect = logsumexp(vector)
    direct = np.log(np.sum(np.exp(vector)))
    assert np.isclose(indirect, direct)

    for vector in np.random.normal(size=(10, 5)):
        indirect = logsumexp(vector)
        direct = np.log(np.sum(np.exp(vector)))
        assert np.isclose(indirect, direct)


def _test_that_logsumexp_returns_the_right_shapes_and_dtypes():

    for rank in [0, 1, 2, 3, 4]:
        shape = np.random.randint(1, 5, size=rank)
        vector = np.random.normal(size=shape)
        result = logsumexp(vector)
        assert result.dtype == vector.dtype
        assert result.shape == ()

    for rank in [0, 1, 2, 3, 4]:
        shape = np.random.randint(1, 5, size=rank)
        vector = np.random.normal(size=shape)
        result = logsumexp(vector, keepdims=True)
        assert result.dtype == vector.dtype
        assert np.all(result.shape == np.ones(rank))


if __name__ == "__main__":

    from tqdm import tqdm
    from scipy import stats
    from matplotlib import pyplot as plt

    nobs = 2000
    dim = 50
    true_mean = np.random.normal(size=dim)
    scale = np.random.normal(size=(dim, dim))
    true_cov = scale @ scale.T
    observations = stats.multivariate_normal.rvs(
        mean=true_mean,
        cov=true_cov,
        size=nobs,
        )
    moments = MixedGaussianModel(np.zeros(dim), np.eye(dim), df=0.5)

    logps = []
    for x in tqdm(observations):
        pure = moments.compute_all_log_likes(x)
        mixed = moments.compute_mixed_log_like(x)
        logps.append((*pure, mixed))
        moments.update_with_single_observation(x)

    cumulatives = np.cumsum(logps, axis=0)
    span = 1 + np.arange(len(cumulatives))
    plt.figure(figsize=(12, 8))
    for idx, label in enumerate(moments.models + ["mixed"]):
        plt.plot(cumulatives[:, idx] / span, label=label)
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("$\\frac{1}{t} \\sum_{s=0}^{t} \\log "
               "p(x_s \\, | \\, x_0, \\ldots, x_{s - 1})$")
    plt.show()
