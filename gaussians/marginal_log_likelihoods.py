import numpy as np
from scipy import special


def combine_stats(mean0, cov0, size0, mean1, cov1, size1):
    weight1 = size1 / (size1 + size0)
    weight0 = size0 / (size1 + size0)
    meandev = np.outer(mean0 - mean1, mean0 - mean1)
    mean = weight0*mean0 + weight1*mean1
    cov = weight0*cov0 + weight1*cov1 + weight0*weight1*meandev
    size = size0 + size1
    return mean, cov, size


def correlated_loglikes(mean, cov, length, mu0, cov0, length0, df0):
    _, covN, lengthN = combine_stats(mu0, cov0, length0, mean, cov, length)
    dfN = df0 + length
    _, logdetscatter0 = np.linalg.slogdet(length0 * cov0)
    _, logdetscatterN = np.linalg.slogdet(lengthN * covN)
    dim, = mean.shape
    return (
        + special.multigammaln((dfN + dim - 1) / 2, dim)
        - special.multigammaln((df0 + dim - 1) / 2, dim)
        + 0.5 * dim * np.log(length0 / lengthN)
        + 0.5 * (df0 + dim - 1) * logdetscatter0
        - 0.5 * (dfN + dim - 1) * logdetscatterN
        - 0.5 * dim * length * np.log(np.pi)
    )


def uncorrelated_loglikes(mean, cov, length, mu0, cov0, length0, df0):
    _, covN, lengthN = combine_stats(mu0, cov0, length0, mean, cov, length)
    dfN = df0 + length
    logdetscatter0 = np.log(length0 * cov0.diagonal()).sum()
    logdetscatterN = np.log(lengthN * covN.diagonal()).sum()
    dim = mu0.shape[-1]
    return (
        + dim * special.gammaln(dfN / 2)
        - dim * special.gammaln(df0 / 2)
        + 0.5 * dim * np.log(length0 / lengthN)
        + 0.5 * df0 * logdetscatter0
        - 0.5 * dfN * logdetscatterN
        - dim * 0.5 * length * np.log(np.pi)
    )


def marginal_logp_shared_variance(statsx, statsy, statsx0, statsy0, df0):

    _, covx0, lengthx0 = statsx0
    _, covy0, lengthy0 = statsy0
    _, _, lengthx = statsx
    _, _, lengthy = statsy
    dim, _ = covx0.shape

    dfNM = df0 + lengthx + lengthy
    _, covxN, lengthxN = combine_stats(*statsx, *statsx0)
    _, covyM, lengthyM = combine_stats(*statsy, *statsy0)
    scatter0 = lengthx0*covx0 + lengthy0*covy0
    scatterNM = lengthxN*covxN + lengthyM*covyM
    _, logdetscatter0 = np.log(scatter0.diagonal()).sum()
    _, logdetscatterNM = np.log(scatterNM.diagonal()).sum()

    return (
        + dim * special.multigammaln(dfNM / 2)
        - dim * special.multigammaln(df0 / 2)
        + 0.5 * dim * np.log(lengthx0 / lengthxN)
        + 0.5 * dim * np.log(lengthy0 / lengthyM)
        + 0.5 * df0 * logdetscatter0
        - 0.5 * dfNM * logdetscatterNM
        - 0.5 * dim * (lengthx + lengthy) * np.log(np.pi)
        )


def marginal_logp_shared_covariance(statsx, statsy, statsx0, statsy0, df0):

    _, covx0, lengthx0 = statsx0
    _, covy0, lengthy0 = statsy0
    _, _, lengthx = statsx
    _, _, lengthy = statsy
    dim, _ = covx0.shape

    dfNM = df0 + lengthx + lengthy
    _, covxN, lengthxN = combine_stats(*statsx, *statsx0)
    _, covyM, lengthyM = combine_stats(*statsy, *statsy0)
    _, logdetscatter0 = np.linalg.slogdet(lengthx0*covx0 + lengthy0*covy0)
    _, logdetscatterNM = np.linalg.slogdet(lengthxN*covxN + lengthyM*covyM)

    return (
        + special.multigammaln((dfNM + dim - 1) / 2, dim)
        - special.multigammaln((df0 + dim - 1) / 2, dim)
        + 0.5 * dim * np.log(lengthx0 / lengthxN)
        + 0.5 * dim * np.log(lengthy0 / lengthyM)
        + 0.5 * (df0 + dim - 1) * logdetscatter0
        - 0.5 * (dfNM + dim - 1) * logdetscatterNM
        - 0.5 * dim * (lengthx + lengthy) * np.log(np.pi)
        )


def marginal_logp_separate_covariances(statsx, statsy, statsx0, statsy0, df0):

    slogpx = correlated_loglikes(*statsx, *statsx0, df0)
    slogpy = correlated_loglikes(*statsy, *statsy0, df0)

    return slogpx + slogpy


def marginal_logp_separate_variances(statsx, statsy, statsx0, statsy0, df0):

    slogpx = uncorrelated_loglikes(*statsx, *statsx0, df0)
    slogpy = uncorrelated_loglikes(*statsy, *statsy0, df0)

    return slogpx + slogpy


if __name__ == "__main__":

    from scipy.stats import wishart
    from scipy.stats import multivariate_normal
    from matplotlib import pyplot as plt

    def compute_stats(sample):
        mean = np.mean(sample, axis=0)
        cov = np.cov(sample.T, ddof=0)
        size = len(sample)
        return mean, cov, size

    plt.close("all")
    length = 2000
    figure, axlist = plt.subplots(
        nrows=3,
        figsize=(12, 8),
        # sharey=True,
        sharex=True,
        )

    maxdim = 100
    fullmeanx = np.random.normal(size=maxdim)
    fullmeany = np.random.normal(size=maxdim)
    fullcovx = wishart.rvs(maxdim, np.eye(maxdim))
    fullcovy = wishart.rvs(maxdim, np.eye(maxdim))
    fullsamplex = multivariate_normal.rvs(fullmeanx, fullcovx, size=length)
    fullsampley = multivariate_normal.rvs(fullmeany, fullcovy, size=length)

    fullmeanx0 = np.random.normal(size=maxdim)
    fullmeany0 = np.random.normal(size=maxdim)
    fullcovx0 = wishart.rvs(maxdim, np.eye(maxdim))
    fullcovy0 = wishart.rvs(maxdim, np.eye(maxdim))
    lengthx0 = 1.0
    lengthy0 = 1.0

    df0 = maxdim

    for axes, dim in zip(axlist, [30, 100, 300]):

        priorx = fullmeanx0[:dim], fullcovx0[:dim, :dim], lengthx0
        priory = fullmeany0[:dim], fullcovy0[:dim, :dim], lengthy0
        
        snippet_lengths = np.arange(10, length, 10)
        logps1 = []
        logps2 = []
        logps3 = []
        for t in snippet_lengths:
            empx = compute_stats(fullsamplex[:t + 1, :dim])
            empy = compute_stats(fullsampley[:t + 1, :dim])
            logp1 = marginal_logp_separate_variances(empx, empy, priorx, priory, df0)
            logp2 = marginal_logp_shared_covariance(empx, empy, priorx, priory, df0)
            logp3 = marginal_logp_separate_covariances(empx, empy, priorx, priory, df0)
            logps1.append(logp1)
            logps2.append(logp2)
            logps3.append(logp3)

        span = 1 + np.arange(length)
        # axes.plot(logps1 / span, "-", alpha=0.8, label="variances only")
        axes.plot(snippet_lengths, logps2 / snippet_lengths, "-",
                  alpha=0.8, label="one covariance matrix")
        axes.plot(snippet_lengths, logps3 / snippet_lengths, "-",
                  alpha=0.8, label="two covariance matrices")
        axes.set_ylabel("$\\frac{1}{N} \\log p(x_1, \\ldots, x_N)$")
        axes.set_xlabel("$N$")
        axes.set_title("Relative log-likelihoods, dim=%s" % dim)
        axes.legend()
    
    figure.tight_layout()
    figure.show()