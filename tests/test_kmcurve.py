import unittest

import numpy as np

from idmlaser.kmcurve import pdsod
from idmlaser.kmcurve import pysod


class TestPysod(unittest.TestCase):
    def test_pysod(self):
        ages_years = np.random.randint(100, size=1024, dtype=np.int32)
        max_year = 100

        result = pysod(ages_years, max_year, np.uint32(np.random.randint(2**32)))
        assert all(result >= ages_years), f"pysod should be >= current age (in years) {result=}, {ages_years=}"
        assert all(result <= max_year), f"pysod should be <= max year {result=}, {max_year=}"


class TestPdsod(unittest.TestCase):
    def test_pdsod(self):
        ages_years = np.random.randint(100, size=1024, dtype=np.int32)
        ages_days = ages_years + np.random.randint(365, size=ages_years.shape[0], dtype=ages_years.dtype)
        max_year = 100

        result = pdsod(ages_days, max_year)
        assert all(result >= ages_days), f"pdsod should be >= current age (in days) {result=}, {ages_days=}"
        assert all(result <= ((max_year + 1) * 365)), f"pdsod should be <= max year + 1 (in days) {result=}, {max_year=}"

    def test_pdsod_max(self):
        ages_days = np.array([100 * 365 + 364], dtype=np.int32)
        max_year = 100

        result = pdsod(ages_days, max_year)
        assert all(result >= ages_days), f"pdsod should be >= current age (in days) {result=}, {ages_days=}"
        assert all(result <= ((max_year + 1) * 365)), f"pdsod should be <= max year + 1 (in days) {result=}, {max_year=}"


if __name__ == "__main__":
    unittest.main()
