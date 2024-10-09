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


if __name__ == "__main__":

    test_saving_and_loading_of_discriminator()
    test_saving_and_restoring_of_discriminator()
    