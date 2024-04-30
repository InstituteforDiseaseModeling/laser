"""Tests for the utils module."""

import unittest

import numpy as np
import pytest

from idmlaser.utils import PriorityQueue
from idmlaser.utils import daily_births_deaths_from_annual
from idmlaser.utils import pop_from_cbr_and_mortality


class TestPriorityQueue(unittest.TestCase):
    def setUp(self):
        self.pq = PriorityQueue(3)

    def test_push_pop(self):
        self.pq.push(1, 2)
        self.pq.push(2, 1)
        self.pq.push(3, 3)

        assert self.pq.pop() == (2, 1)
        assert self.pq.pop() == (1, 2)
        assert self.pq.pop() == (3, 3)

    def test_push_pop_random(self):
        values = np.random.randint(0, 100, 1024)
        self.pq = PriorityQueue(len(values))
        for i in values:
            self.pq.push(i, i)
        minimum = 0
        while len(self.pq) > 0:
            value, _ = self.pq.pop()
            assert value >= minimum
            minimum = value

    def test_peek(self):
        self.pq.push(1, 2)
        self.pq.push(2, 1)
        self.pq.push(3, 3)

        assert self.pq.peek() == (2, 1)

    def test_empty_peek(self):
        with pytest.raises(IndexError):
            _ = self.pq.peek()

    def test_peek_random(self):
        values = np.random.randint(0, 100, 1024)
        self.pq = PriorityQueue(len(values))
        for i in values:
            self.pq.push(i, i)
        minimum = values.min()
        assert self.pq.peek() == (minimum, minimum)

    def test_empty_pop(self):
        with pytest.raises(IndexError):
            self.pq.pop()

    def test_full_push(self):
        self.pq.push(1, 2)
        self.pq.push(2, 1)
        self.pq.push(3, 3)

        with pytest.raises(IndexError):
            self.pq.push(4, 4)


class TestPopFromCbrAndMortality(unittest.TestCase):
    INITIAL = 100_000
    CBR = 25  # 25/1000 = 2.5% growth per annum
    MORTALITY = 17  # 17/1000 = 1.7% mortality per annum ~60 years life expectancy
    NYEARS = 50

    def test_pop_from_cbr_constant_and_mortality_constant(self):
        births, deaths, population = pop_from_cbr_and_mortality(self.INITIAL, self.CBR, self.MORTALITY, self.NYEARS)

        assert len(births) == self.NYEARS
        assert len(deaths) == self.NYEARS
        assert len(population) == (self.NYEARS + 1)

        for i in range(len(population) - 1):
            assert population[i + 1] == population[i] + births[i] - deaths[i]

        expected_population = np.uint32(np.round(self.INITIAL * (1 + (self.CBR - self.MORTALITY) / 1000) ** self.NYEARS))
        assert abs(1 - population[-1] / expected_population) < 0.01, f"Expected {expected_population}, got {population[-1]}"

        return

    def test_pop_from_cbr_varying_and_mortality_constant(self):
        cbr = np.random.normal(self.CBR, 3, self.NYEARS)

        births, deaths, population = pop_from_cbr_and_mortality(self.INITIAL, cbr, self.MORTALITY, self.NYEARS)

        assert len(births) == self.NYEARS
        assert len(deaths) == self.NYEARS
        assert len(population) == (self.NYEARS + 1)

        for i in range(len(population) - 1):
            assert population[i + 1] == population[i] + births[i] - deaths[i]

        expected_population = np.uint32(np.round(self.INITIAL * (1 + (cbr.mean() - self.MORTALITY) / 1000) ** self.NYEARS))
        assert abs(1 - population[-1] / expected_population) < 0.01, f"Expected {expected_population}, got {population[-1]}"

        return

    def test_pop_from_cbr_constant_and_mortality_varying(self):
        mortality = np.random.normal(self.MORTALITY, 2, self.NYEARS)

        births, deaths, population = pop_from_cbr_and_mortality(self.INITIAL, self.CBR, mortality, self.NYEARS)

        assert len(births) == self.NYEARS
        assert len(deaths) == self.NYEARS
        assert len(population) == (self.NYEARS + 1)

        for i in range(len(population) - 1):
            assert population[i + 1] == population[i] + births[i] - deaths[i]

        expected_population = np.uint32(np.round(self.INITIAL * (1 + (self.CBR - mortality.mean()) / 1000) ** self.NYEARS))
        assert abs(1 - population[-1] / expected_population) < 0.01, f"Expected {expected_population}, got {population[-1]}"

        return

    def test_pop_from_cbr_varying_and_mortality_varying(self):
        cbr = np.random.normal(self.CBR, 3, self.NYEARS)
        mortality = np.random.normal(self.MORTALITY, 2, self.NYEARS)

        births, deaths, population = pop_from_cbr_and_mortality(self.INITIAL, cbr, mortality, self.NYEARS)

        assert len(births) == self.NYEARS
        assert len(deaths) == self.NYEARS
        assert len(population) == (self.NYEARS + 1)

        for i in range(len(population) - 1):
            assert population[i + 1] == population[i] + births[i] - deaths[i]

        expected_population = np.uint32(np.round(self.INITIAL * (1 + (cbr.mean() - mortality.mean()) / 1000) ** self.NYEARS))
        assert abs(1 - population[-1] / expected_population) < 0.01, f"Expected {expected_population}, got {population[-1]}"

        return


class TestDailyBirthsDeathsFromAnnual(unittest.TestCase):
    NYEARS = 10

    def test_from_fixed_values(self):
        annual_births = np.full(self.NYEARS, 730)
        annual_deaths = np.full(self.NYEARS, 365)

        daily_births, daily_deaths = daily_births_deaths_from_annual(annual_births, annual_deaths)

        # Assert that the actual daily births and deaths match the expected values
        assert np.all(daily_births == 2), f"Expected 2, got {daily_births}"
        assert np.all(daily_deaths == 1), f"Expected 1, got {daily_deaths}"
        assert daily_births.sum() == (self.NYEARS * 730), f"Expected {self.NYEARS * 730}, got {daily_births.sum()}"
        assert daily_deaths.sum() == (self.NYEARS * 365), f"Expected {self.NYEARS * 365}, got {daily_deaths.sum()}"

    def test_from_stochastic_values(self):
        annual_births = np.random.poisson(730, self.NYEARS)
        annual_deaths = np.random.poisson(365, self.NYEARS)

        daily_births, daily_deaths = daily_births_deaths_from_annual(annual_births, annual_deaths)

        # Assert that the actual daily births and deaths match the expected values
        assert np.all(daily_births >= 0), f"Expected >= 0, got {daily_births}"
        assert np.all(daily_deaths >= 0), f"Expected >= 0, got {daily_deaths}"
        assert daily_births.sum() == annual_births.sum(), f"Expected {annual_births.sum()}, got {daily_births.sum()}"
        assert daily_deaths.sum() == annual_deaths.sum(), f"Expected {annual_deaths.sum()}, got {daily_deaths.sum()}"
        self.assertAlmostEqual(daily_births.mean(), annual_births.mean() / 365, places=4)  # noqa: PT009
        self.assertAlmostEqual(daily_deaths.mean(), annual_deaths.mean() / 365, places=4)  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
