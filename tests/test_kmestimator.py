import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kstest

from laser_core.demographics import AliasedDistribution
from laser_core.demographics import KaplanMeierEstimator
from laser_core.demographics import load_pyramid_csv


class TestKaplanMeierEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (Path(__file__).parent / "data" / "us-life-tables-nvs-2003.csv").open("r") as file:
            cls.cumulative_deaths = np.insert(np.loadtxt(file, delimiter=",", usecols=1).astype(np.uint32), 0, 0)
        cls.estimator = KaplanMeierEstimator(cls.cumulative_deaths)

        return

    def test_predict_year_of_death_limits_with_default(self):
        ages_years = np.random.randint(100, size=1024, dtype=np.int32)
        result = self.estimator.predict_year_of_death(ages_years)
        assert all(result >= ages_years), f"pysod should be >= current age (in years) {result=}, {ages_years=}"
        assert all(result <= 100), f"pysod should be <= 100 ({result.max()=})"

        return

    def test_predict_year_of_death_limits_with_maximum(self):
        max_year = 80
        ages_years = np.random.randint(max_year, size=1024, dtype=np.int32)
        result = self.estimator.predict_year_of_death(ages_years, max_year)
        assert all(result >= ages_years), f"pysod should be >= current age (in years) {result=}, {ages_years=}"
        assert all(result <= max_year), f"pysod should be <= max year ({max_year=}, {result.max()=})"

        return

    def test_predict_year_of_death_kstest(self):
        ages_years = np.zeros(100_000, dtype=np.int32)
        predictions = self.estimator.predict_year_of_death(ages_years, 100)
        counts = np.zeros(predictions.max() + 1, dtype=np.int32)
        np.add.at(counts, predictions, 1)
        f_of_x = np.insert(np.cumsum(counts), 0, 0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="ks_2samp: Exact calculation unsuccessful.*",
                category=RuntimeWarning,
            )
            test = kstest(f_of_x, self.cumulative_deaths)
        assert test.pvalue > 0.80, f"Kolmogorov-Smirnov test failed {test.pvalue=}"

        return

    def test_predict_age_at_death_limits_default(self):
        ages_years = np.random.randint(100, size=1024, dtype=np.int32)
        ages_days = ages_years * 365 + np.random.randint(365, size=ages_years.shape[0], dtype=ages_years.dtype)
        result = self.estimator.predict_age_at_death(ages_days)
        assert all(result >= ages_days), f"predicted age at death should be >= current age (in days) {result=}, {ages_days=}"
        assert all(result < ((100 + 1) * 365)), f"predicted age at death should be < (100 + 1) (in days) ({result.max()=})"

        return

    def test_predict_age_at_death_limits_with_maximum(self):
        max_year = 80
        ages_years = np.random.randint(max_year, size=1024, dtype=np.int32)
        ages_days = ages_years * 365 + np.random.randint(365, size=ages_years.shape[0], dtype=ages_years.dtype)

        result = self.estimator.predict_age_at_death(ages_days, max_year)
        assert all(result >= ages_days), f"predicted age at death should be >= current age (in days) {result=}, {ages_days=}"
        assert all(
            result < ((max_year + 1) * 365)
        ), f"predicted age at death should be < (max_year + 1) (in days) ({max_year=}, {result.max()=})"

        return

    def test_predict_age_at_death_kstest(self):
        # We could just do a uniform draw from 0-100 years, but that would be boring.
        pyramid = load_pyramid_csv(Path(__file__).parent / "data" / "us-pyramid-2023.csv")
        both = pyramid[:, 2] + pyramid[:, 3]  # males and females combined
        ad = AliasedDistribution(both)
        ages_years = ad.sample(1_000_000)
        ages_days = ages_years * 365 + np.random.randint(365, size=ages_years.shape[0], dtype=ages_years.dtype)
        failed = 0
        for _ in range(4):
            predictions = self.estimator.predict_age_at_death(ages_days, 100)
            predicted_years = np.floor_divide(predictions, 365, dtype=np.int32)  # convert age (days) at death to age (years) at death
            assert predicted_years.max() <= 100, f"predicted years of death should be <= 100 ({predicted_years.max()=})"
            # for each starting age, check against the (remaining) survival curve
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="ks_2samp: Exact calculation unsuccessful.*",
                    category=RuntimeWarning,
                )
                for age in range(ages_years.max() + 1):
                    individuals = np.nonzero(ages_years == age)[0]  # indices of individuals who are age `age`
                    counts = np.zeros(len(self.cumulative_deaths) - 1, dtype=np.int32)  # zeroed array
                    np.add.at(counts, predicted_years[individuals], 1)  # histogram of deaths at age i
                    f_of_x = np.insert(np.cumsum(counts), 0, 0)  # insert a 0 to match the estimator.cumulative deaths and simplify the math
                    f_of_x = f_of_x[age + 1 :] - f_of_x[age]  # remaining deaths at year >= age
                    g_of_x = self.cumulative_deaths[age + 1 :] - self.cumulative_deaths[age]  # survival curve for years >= age
                    factor = f_of_x[-1] / g_of_x[-1]  # comparison factor
                    g_of_x = (factor * g_of_x).astype(g_of_x.dtype)  # use same total number of deaths
                    test = kstest(f_of_x, g_of_x)
                    if test.pvalue < 0.80:
                        print(f"{test.pvalue=}")
                        failed += 1
                    # assert test.pvalue >= 0.80, f"Kolmogorov-Smirnov test failed for {age=} ({test.pvalue=})"
        assert failed < 4, f"All ({failed}) Kolmogorov-Smirnov tests failed."

        return

    def test_predict_year_of_death_with_types(self):
        for type in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
            max_year = 100
            ages_years = np.random.randint(max_year + 1, size=1024).astype(type)
            result = self.estimator.predict_year_of_death(ages_years, max_year=max_year)
            assert result.dtype == np.uint16, f"Expected {type=}, got {result.dtype=}"

        return

    def test_predict_age_at_death_with_types(self):
        for type in [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
            # Use 88 as maximum age to avoid overflow with int16
            max_year = 88 if type == np.int16 else 100
            ages_years = np.random.randint(max_year + 1, size=1024).astype(type)
            ages_days = ages_years * 365 + np.random.randint(365, size=ages_years.shape[0]).astype(ages_years.dtype)
            result = self.estimator.predict_age_at_death(ages_days, max_year=max_year)
            assert result.dtype == type, f"Expected {type=}, got {result.dtype=}"

        return


if __name__ == "__main__":
    unittest.main()
