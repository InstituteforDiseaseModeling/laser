"""Test function in the pyramid module."""

import unittest

import numpy as np

from idmlaser.pyramid import AliasedDistribution


class AliasedDistributionTests(unittest.TestCase):
    def test_init(self):
        counts = [1, 2, 3, 4, 5]
        ad = AliasedDistribution(counts)
        assert ad.total == 15
        assert ad.alias.shape == (5,)
        assert ad.probs.shape == (5,)

    def test_sample(self):
        counts = [1, 2, 3, 4, 5]
        ad = AliasedDistribution(counts)
        s = ad.sample()
        assert s >= 0
        assert s < 5

    def test_sample_multiple(self):
        counts = [1, 2, 3, 4, 5]
        ad = AliasedDistribution(counts)
        s = ad.sample(10)
        assert s.shape == (10,)
        assert (s >= 0).all()
        assert (s < 5).all()

    def test_sample_prng(self):
        counts = [1, 2, 3, 4, 5]
        ad1 = AliasedDistribution(counts, prng=np.random.default_rng(42))
        s1 = ad1.sample()
        assert s1 >= 0
        assert s1 < 5
        ad2 = AliasedDistribution(counts, prng=np.random.default_rng(42))
        s2 = ad2.sample()
        assert s2 == s1

    def test_sample_multiple_prng(self):
        counts = [1, 2, 3, 4, 5]
        ad1 = AliasedDistribution(counts, prng=np.random.default_rng(42))
        s1 = ad1.sample(10)
        assert s1.shape == (10,)
        assert (s1 >= 0).all()
        assert (s1 < 5).all()
        ad2 = AliasedDistribution(counts, prng=np.random.default_rng(42))
        s2 = ad2.sample(10)
        assert np.array_equal(s2, s1)


if __name__ == "__main__":
    unittest.main()
