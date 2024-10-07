import numpy as np
from scipy import stats

from marginal_log_likelihoods import correlated_loglikes
from marginal_log_likelihoods import uncorrelated_loglikes
from moments_tracker import MomentsTracker


def test_marginal_log_likelihood_formulas():

    length = np.random.randint(2, 100)
    dim = np.random.randint(1, 30)
    sample = np.random.normal(size=(length, dim)) @ np.random.normal(size=(dim, dim))

    tracker = MomentsTracker.random(dim)
    empirical = MomentsTracker.fromdata(sample)

    df0 = np.random.gamma(1.0)
    correlated = correlated_loglikes(*empirical, *tracker, df0)
    uncorrelated = uncorrelated_loglikes(*empirical, *tracker, df0)

    df = np.copy(df0)
    correlated_logps = []
    uncorrelated_logps = []

    for x in sample:

        # compute multivariate t log-density
        shape = (tracker.count + 1) / df * tracker.cov
        logp = stats.multivariate_t.logpdf(x, loc=tracker.mean, shape=shape, df=df)
        correlated_logps.append(logp)

        # compute univariate t log-densities
        var = (tracker.count + 1) / df * tracker.cov.diagonal()
        tlogp = stats.t.logpdf(x, loc=tracker.mean, df=df, scale=np.sqrt(var)).sum()
        uncorrelated_logps.append(tlogp)

        tracker.update_with_single(x)
        df += 1  # the degrees of freedom is not part of the moments tracker

    print("Uncorrelated logp (N=%s, D=%s):" % (length, dim))
    print(np.sum(uncorrelated_logps, axis=0))
    print(uncorrelated)
    print()
    print("Correlated logp (N=%s, D=%s):" % (length, dim))
    print(np.sum(correlated_logps, axis=0))
    print(correlated)
    print()
    logprobs = np.array([uncorrelated, correlated])
    probs = np.exp(logprobs - logprobs.max())
    probs /= probs.sum()
    print("Prob uncorrelated: %.5f" % probs[0])
    print("Prob correlated: %.5f" % probs[1])
    print()

    # check that the pre-computed sum matches the sequentially updated:
    assert np.allclose(np.sum(uncorrelated_logps, axis=0), uncorrelated)
    assert np.allclose(np.sum(correlated_logps, axis=0), correlated)


if __name__ == "__main__":

    test_marginal_log_likelihood_formulas()