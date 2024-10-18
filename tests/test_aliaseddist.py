import unittest
from pathlib import Path

import numpy as np
from scipy.stats import kstest

from laser_core.demographics import AliasedDistribution
from laser_core.demographics import load_pyramid_csv


class TestAliasedDistribution(unittest.TestCase):
    def test_aliased_distribution(self):
        pyramid = load_pyramid_csv(Path(__file__).parent / "data" / "us-pyramid-2023.csv", quiet=True)
        ad = AliasedDistribution(pyramid[:, 4])  # males and females
        nsamples = 1_000_000
        samples = ad.sample(nsamples)
        counts = np.zeros(samples.max() + 1, dtype=np.int32)
        np.add.at(counts, samples, 1)
        f_of_x = counts.cumsum()
        g_of_x = pyramid[:, 4].cumsum()
        total = pyramid[:, 4].sum()
        g_of_x = (g_of_x / (total / nsamples)).astype(g_of_x.dtype)
        test = kstest(f_of_x, g_of_x)
        assert test.pvalue > 0.05, f"Kolmogorov-Smirnov test failed {test=}"
        return

    def test_load_pyramid_csv(self):
        pyramid = load_pyramid_csv(Path(__file__).parent / "data" / "us-pyramid-2023.csv", quiet=True)
        assert pyramid.shape == (21, 5), f"Expected pyramid shape == (21, 5) got {pyramid.shape=}"
        assert pyramid[0, 0] == 0, f"Expected pyramid[0, 0] == 0 got {pyramid[0, 0]=}"
        assert pyramid[0, 1] == 4, f"Expected pyramid[0, 1] == 4 got {pyramid[0, 1]=}"
        assert pyramid[-1, 0] == 100, f"Expected pyramid[-1, 0] == 100 got {pyramid[-1, 0]=}"
        assert pyramid[-1, 1] == 100, f"Expected pyramid[-1, 1] == 100 got {pyramid[-1, 1]=}"
        assert np.all(
            (pyramid[:, 2] + pyramid[:, 3]) == pyramid[:, 4]
        ), f"Expected sum of male ({pyramid[:, 2].sum()}) + female ({pyramid[:, 3].sum()}) == total population {pyramid[:, 4].sum()=}"


if __name__ == "__main__":
    unittest.main()
