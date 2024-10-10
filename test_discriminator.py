import os
from tempfile import TemporaryDirectory
import numpy as np
from discriminator import BiGaussianDiscriminator
from gaussians.moments_tracker import MomentsTracker


def test_saving_and_loading_of_discriminator():
    xstats = MomentsTracker.fromdata(np.random.normal(size=(13, 3)))
    ystats = MomentsTracker.fromdata(np.random.normal(size=(13, 3)))
    original = BiGaussianDiscriminator()
    original.fit_with_moments(xstats, ystats)
    with TemporaryDirectory() as tempdir:
        outpath = os.path.join(tempdir, "params.npz")
        original.save(outpath)
        restored = BiGaussianDiscriminator()
        restored.load(outpath)
    assert np.allclose(restored.dist_pos.mean, original.dist_pos.mean)
    assert np.allclose(restored.dist_pos.cov, original.dist_pos.cov)
    assert np.allclose(restored.dist_neg.mean, original.dist_neg.mean)
    assert np.allclose(restored.dist_neg.cov, original.dist_neg.cov)


def test_saving_and_restoring_of_discriminator():
    xstats = MomentsTracker.fromdata(np.random.normal(size=(13, 3)))
    ystats = MomentsTracker.fromdata(np.random.normal(size=(13, 3)))
    original = BiGaussianDiscriminator()
    original.fit_with_moments(xstats, ystats)
    with TemporaryDirectory() as tempdir:
        outpath = os.path.join(tempdir, "params.npz")
        original.save(outpath)
        restored = BiGaussianDiscriminator.fromsaved(outpath)
    assert np.allclose(restored.dist_pos.mean, original.dist_pos.mean)
    assert np.allclose(restored.dist_pos.cov, original.dist_pos.cov)
    assert np.allclose(restored.dist_neg.mean, original.dist_neg.mean)
    assert np.allclose(restored.dist_neg.cov, original.dist_neg.cov)


def test_that_discriminator_correctly_distinguishes_separable_classes():
    length1, length2 = np.random.randint(1000, 2000, size=2)
    pos_data = np.random.normal(size=(length1, 2)) + 5.0
    neg_data = np.random.normal(size=(length2, 2)) - 5.0
    pos_stats = MomentsTracker.fromdata(pos_data)
    neg_stats = MomentsTracker.fromdata(neg_data)
    discriminator = BiGaussianDiscriminator()
    discriminator.fit_with_moments(pos_stats, neg_stats)
    true_pos_rate = np.mean(discriminator(pos_data) > 0)
    assert true_pos_rate > 0.99
    true_neg_rate = np.mean(discriminator(neg_data) < 0)
    assert true_neg_rate > 0.99


if __name__ == "__main__":

    test_saving_and_loading_of_discriminator()
    test_saving_and_restoring_of_discriminator()
    test_that_discriminator_correctly_distinguishes_separable_classes()

