import os
from tempfile import TemporaryDirectory
import numpy as np

from gaussians.moments_tracker import MomentsTracker
from gaussians.moments_tracker import mixed_covariance
from gaussians.moments_tracker import combine
from gaussians.moments_tracker import combine_covariance_only


def test_combine_moment_trackers():

    dim = np.random.randint(1, 5)
    sizes = np.random.randint(dim, 10 + dim, size=3)
    samples = [np.random.normal(size=(n, dim)) for n in sizes]

    means = [np.mean(x, axis=0) for x in samples]
    covs = [np.cov(x.T, ddof=0) for x in samples]
    counts = [len(x) for x in samples]
    moments = [MomentsTracker(*args) for args in zip(means, covs, counts)]
    combined = combine(moments)

    sample = np.concatenate(samples, axis=0)
    mean = np.mean(sample, axis=0)
    cov = np.cov(sample.T, ddof=0)
    count = len(sample)

    assert np.allclose(combined.mean, mean)
    assert np.allclose(combined.cov, cov)
    assert np.isclose(combined.count, count)


def test_moments_tracker_with_unweighted_priors():

    length = np.random.randint(6, 25)
    dim = np.random.randint(1, 5)
    sample = np.random.normal(size=(length, dim))

    # use stepwise updates:
    m1 = MomentsTracker(np.zeros(dim), np.eye(dim), count=0)
    for observation in sample:
        m1.update_with_single(observation)
    
    # use a single large update:
    m2 = MomentsTracker(np.zeros(dim), np.eye(dim), count=0)
    m2.update_with_batch(sample)

    k = np.random.randint(1, 5)  # num sections
    slengths = 1 + np.random.multinomial(n=length - k, pvals=np.ones(k) / k)
    assert len(slengths) > 0
    assert all(slengths > 0)
    m3 = MomentsTracker(np.zeros(dim), np.eye(dim), count=0)
    batches = np.split(sample, np.cumsum(slengths[:-1]))
    assert all(b.shape == (N, dim) for N, b in zip(slengths, batches))
    for batch in batches:
        m3.update_with_batch(batch)

    numpy_mean = np.mean(sample, axis=0)
    numpy_cov = np.cov(sample.T, ddof=0)

    assert np.allclose(m1.count, length)
    assert np.allclose(m1.mean, numpy_mean)
    assert np.allclose(m1.cov, numpy_cov)

    assert np.allclose(m2.count, length)
    assert np.allclose(m2.mean, numpy_mean)
    assert np.allclose(m2.cov, numpy_cov)

    assert np.allclose(m3.count, length)
    assert np.allclose(m3.mean, numpy_mean)
    assert np.allclose(m3.cov, numpy_cov)


def test_moments_tracker_with_weighted_priors():

    length1 = np.random.randint(6, 25)
    dim = np.random.randint(1, 5)

    # create a moments tracker:
    mean0 = np.zeros(dim)
    cov0 = np.eye(dim)
    moments = MomentsTracker(mean0, cov0, count=0)
    # check that it is initialized as requested:
    assert np.allclose(moments.mean, mean0)
    assert np.allclose(moments.cov, cov0)
    assert np.allclose(moments.count, 0)

    # create a random sample:
    sample1 = np.random.normal(size=(length1, dim))
    # and compute its moments directly:
    mean1 = np.mean(sample1, axis=0)
    cov1 = np.cov(sample1.T, ddof=0)

    # now ACTUALLY update the tracker:
    moments.update_with_batch(sample1)
    # and check that that results in the right moments:
    assert np.allclose(moments.mean, mean1)
    assert np.allclose(moments.cov, cov1)
    assert np.allclose(moments.count, length1)

    # extend the sample with extra observations:
    length2 = np.random.randint(6, 25)
    sample2 = np.random.normal(size=(length2, dim))
    sample12 = np.concatenate([sample1, sample2])
    # and compute the grand mean and covariance matrix:
    mean12 = np.mean(sample12, axis=0)
    cov12 = np.cov(sample12.T, ddof=0)
    length12 = len(sample12)
    assert length12 == length1 + length2

    # now ACTUALLY update the tracker:
    moments.update_with_batch(sample2)
    # and check that that results in the right moments:
    assert np.allclose(moments.mean, mean12)
    assert np.allclose(moments.cov, cov12)
    assert np.allclose(moments.count, length12)


def test_moments_tracker_reset():

    sample_count = np.random.randint(6, 25)
    dim = np.random.randint(1, 5)
    sample = np.random.normal(size=(sample_count, dim))
    sample_mean = np.mean(sample, axis=0)
    sample_cov = np.cov(sample.T, ddof=0)

    tracker = MomentsTracker(np.zeros(dim), np.eye(dim), count=0)
    assert tracker.count == 0
    assert np.allclose(tracker.mean, np.zeros(dim))
    assert np.allclose(tracker.cov, np.eye(dim))

    tracker.update_with_batch(sample)
    assert tracker.count == sample_count
    assert np.allclose(tracker.mean, sample_mean)
    assert np.allclose(tracker.cov, sample_cov)

    tracker.reset()
    assert tracker.count == 0
    assert np.allclose(tracker.mean, np.zeros(dim))
    assert np.allclose(tracker.cov, np.eye(dim))


def test_moments_tracker_update_with_moments():

    count1, count2 = np.random.randint(6, 25, size=2)
    dim = np.random.randint(1, 5)

    sample1 = np.random.normal(size=(count1, dim))
    mean1 = np.mean(sample1, axis=0)
    cov1 = np.cov(sample1.T, ddof=0)
    
    sample2 = np.random.normal(size=(count2, dim))
    mean2 = np.mean(sample2, axis=0)
    cov2 = np.cov(sample2.T, ddof=0)
    
    full_sample = np.concatenate([sample1, sample2], axis=0)
    full_mean = np.mean(full_sample, axis=0)
    full_cov = np.cov(full_sample.T, ddof=0)
    full_count = len(full_sample)

    first = MomentsTracker(np.zeros(dim), np.eye(dim), count=0)
    second = MomentsTracker(np.zeros(dim), np.eye(dim), count=0)

    first.update_with_batch(sample1)
    assert first.count == count1
    assert np.allclose(first.mean, mean1)
    assert np.allclose(first.cov, cov1)

    second.update_with_batch(sample2)
    assert second.count == count2
    assert np.allclose(second.mean, mean2)
    assert np.allclose(second.cov, cov2)

    first.update_with_moments(*second)
    assert first.count == full_count
    assert np.allclose(first.mean, full_mean)
    assert np.allclose(first.cov, full_cov)


def test_mixed_covariance():
    sample1 = np.random.normal(size=(5, 2))
    sample2 = np.random.normal(size=(7, 2))
    total_cov = np.cov(np.concatenate([sample1, sample2], axis=0).T, ddof=0)
    mean1 = np.mean(sample1, axis=0)
    mean2 = np.mean(sample2, axis=0)
    cov1 = np.cov(sample1.T, ddof=0)
    cov2 = np.cov(sample2.T, ddof=0)
    size1 = len(sample1)
    size2 = len(sample2)
    mixed_cov = mixed_covariance(mean1, cov1, size1, mean2, cov2, size2)
    assert np.allclose(mixed_cov, total_cov)


def test_moments_tracker_saving_and_loading():
    x = np.random.normal(size=(7, 3))
    original = MomentsTracker.fromdata(x)
    with TemporaryDirectory() as tempdir:
        outpath = os.path.join(tempdir, "stats.npz")
        original.save(outpath)
        restored = MomentsTracker.fromfile(outpath)
    assert np.allclose(restored.mean, np.mean(x, axis=0))
    assert np.allclose(restored.cov, np.cov(x.T, ddof=0))
    assert np.isclose(restored.count, len(x))


def test_combine_covariance_only_with_batch_computation():

    dim = np.random.randint(2, 10)
    count1, count2 = np.random.randint(10, 25, size=2)
    sample1 = np.random.normal(size=(count1, dim))
    sample2 = np.random.normal(size=(count2, dim))

    devs1 = sample1 - np.mean(sample1, axis=0, keepdims=True)
    devs2 = sample2 - np.mean(sample2, axis=0, keepdims=True)
    all_devs = np.concatenate([devs1, devs2], axis=0)

    cov1 = devs1.T @ devs1 / count1
    cov2 = devs2.T @ devs2 / count2
    shared_cov = all_devs.T @ all_devs / (count1 + count2)

    assert np.allclose(cov1, np.cov(sample1.T, ddof=0))
    assert np.allclose(cov2, np.cov(sample2.T, ddof=0))

    combined_cov = combine_covariance_only(
        None, cov1, count1,
        None, cov2, count2,
        )

    assert np.allclose(combined_cov, shared_cov)


def test_combine_covariance_only_with_stepwise_computation():

    # NOTE: suppose two distributions P1 and P2 have means mu1 and mu2,
    # and share a single scatter matrix S. We can then as usual update
    # the means using the rules
    #
    #   mu1 <- k1/(k1 + 1)*mu1 + 1/(k1 + 1)*x    if x ~ P1
    #   mu2 <- k2/(k2 + 1)*mu2 + 1/(k2 + 1)*x    if x ~ P2
    #
    # where k1 is the number of observations seen from P1 so far, and k2
    # the number of observations seen from P2 so far.
    #
    # For the shared scatter matrix S, however, we get the update rules
    #
    #   S <- S + k1/(k1 + 1)**2*outer(x - mu1, x - mu1)    if x ~ P1
    #   S <- S + k2/(k2 + 1)**2*outer(x - mu2, x - mu2)    if x ~ P2
    #
    # This is equivalent to writing S as the sum of two scatter matrices,
    # one per distribution component, and estimating those components
    # separately.
    #
    # If we convert these rules into a single update rule for a shared
    # covariance matrix instead of the shared scatter matrix, it becomes
    #
    #   C <- k/(k + 1)*C + k1/((k1 + 1)*(k + 1))*outer(x - mu1, x - mu1)
    #       if x ~ P1
    #   C <- k/(k + 1)*C + k2/((k2 + 1)*(k + 1))*outer(x - mu2, x - mu2)
    #       if x ~ P2
    #
    # where k = k1 + k2 is the total number of observations with which
    # the covariance-matrix estimate has been updated so far.

    dim = np.random.randint(2, 10)
    count1, count2 = np.random.randint(10, 25, size=2)
    sample1 = np.random.normal(size=(count1, dim))
    sample2 = np.random.normal(size=(count2, dim))

    k = 0
    cov = np.eye(dim)
    k1 = 0
    k2 = 0
    mean1 = np.zeros(dim)
    mean2 = np.zeros(dim)
    cov1 = np.eye(dim)
    cov2 = np.eye(dim)

    for x in sample1:
        cov *= k / (k + 1)
        cov += k1 / ((k + 1) * (k1 + 1)) * np.outer(x - mean1, x - mean1)
        cov1 *= k1 / (k1 + 1)
        cov1 += k1 / (k1 + 1) ** 2 * np.outer(x - mean1, x - mean1)
        mean1 *= k1 / (k1 + 1)
        mean1 += 1 / (k1 + 1) * x
        k1 += 1
        k += 1

    for x in sample2:
        cov *= k / (k + 1)
        cov += k2 / ((k + 1) * (k2 + 1)) * np.outer(x - mean2, x - mean2)
        cov2 *= k2 / (k2 + 1)
        cov2 += k2 / (k2 + 1) ** 2 * np.outer(x - mean2, x - mean2)
        mean2 *= k2 / (k2 + 1)
        mean2 += 1 / (k2 + 1) * x
        k2 += 1
        k += 1

    assert np.allclose(np.mean(sample1, axis=0), mean1)
    assert np.allclose(np.mean(sample2, axis=0), mean2)

    assert np.allclose(np.cov(sample1.T, ddof=0), cov1)
    assert np.allclose(np.cov(sample2.T, ddof=0), cov2)

    assert np.isclose(len(sample1), k1)
    assert np.isclose(len(sample2), k2)

    combined_cov = combine_covariance_only(
        None, cov1, k1,
        None, cov2, k2,
        )

    assert np.allclose(combined_cov, cov)


if __name__ == "__main__":

    test_combine_moment_trackers()
    test_moments_tracker_with_unweighted_priors()
    test_moments_tracker_with_weighted_priors()
    test_moments_tracker_reset()
    test_moments_tracker_update_with_moments()
    test_mixed_covariance()
    test_moments_tracker_saving_and_loading()
    test_combine_covariance_only_with_batch_computation()
    test_combine_covariance_only_with_stepwise_computation()

