import numpy as np
from scipy import special

from moments_tracker import mixed_covariance


def corrrelated_loglikes(mean, cov, length, mu0, cov0, length0, df0):
    dim, = mean.shape
    covN = mixed_covariance(mu0, cov0, length0, mean, cov, length)
    return (
        + special.multigammaln((df0 + dim - 1 + length) / 2, dim)
        - special.multigammaln((df0 + dim - 1) / 2, dim)
        - 0.5 * dim * length * np.log(np.pi)
        + 0.5 * dim * (df0 + dim) * np.log(length0)
        - 0.5 * dim * (df0 + dim + length) * np.log((length0 + length))
        + 0.5 * (df0 + dim - 1) * np.linalg.slogdet(cov0).logabsdet
        - 0.5 * (df0 + dim - 1 + length) * np.linalg.slogdet(covN).logabsdet
    )


def uncorrrelated_loglikes(mean, cov, length, mu0, cov0, length0, df0):
    dim = mu0.shape[-1]
    covN = mixed_covariance(mu0, cov0, length0, mean, cov, length)
    return (
        + dim * special.gammaln((df0 + length) / 2)
        - dim * special.gammaln(df0 / 2)
        - dim * 0.5 * length * np.log(np.pi)
        + dim * 0.5 * (df0 + 1) * np.log(length0)
        - dim * 0.5 * (df0 + length + 1) * np.log(length0 + length)
        + 0.5 * df0 * np.log(cov0.diagonal()).sum()
        - 0.5 * (df0 + length) * np.log(covN.diagonal()).sum()
    )
