"""Tests for the utils module."""

import unittest

import numpy as np
import pytest

from idmlaser.utils import GrabBag
from idmlaser.utils import PriorityQueue
from idmlaser.utils import daily_births_deaths_from_annual
from idmlaser.utils import pop_from_cbr_and_mortality
from idmlaser.utils import predicted_day_of_death
from idmlaser.utils import predicted_year_of_death


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


class TestPredictedDateOfDeath(unittest.TestCase):
    def test_predicted_year_of_death_0_default(self):
        age = 0
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age)
            assert age <= yod <= 100, f"Expected {age} <= {yod} <= 100"

    def test_predicted_year_of_death_0_100(self):
        age = 0
        max_year = 100
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert age <= yod <= max_year, f"Expected {age} <= {yod} <= {max_year}"

    def test_predicted_year_of_death_0_89(self):
        age = 0
        max_year = 89
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert age <= yod <= max_year, f"Expected {age} <= {yod} <= {max_year}"

    def test_predicted_year_of_death_30_default(self):
        age = 30
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age)
            assert age <= yod <= 100, f"Expected {age} <= {yod} <= 100"

    def test_predicted_year_of_death_30_100(self):
        age = 30
        max_year = 100
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert age <= yod <= max_year, f"Expected {age} <= {yod} <= {max_year}"

    def test_predicted_year_of_death_30_89(self):
        age = 30
        max_year = 89
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert age <= yod <= max_year, f"Expected {age} <= {yod} <= {max_year}"

    def test_predicted_year_of_death_default_maximum(self):
        age = 100
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age)
            assert yod == 100, f"Expected {yod} == 100"

    def test_predicted_year_of_death_30_maximum(self):
        age = max_year = 30
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert yod == max_year, f"Expected {yod} == {max_year}"

    def test_predicted_day_of_death_0_default(self):
        age = 0
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age)
            assert age < dod < 36865, f"Expected {age} < {dod} < 36,865 (365 * 100 + 365)"

    def test_predicted_day_of_death_0_100(self):
        age = 0
        max_year = 100
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert age < dod < (max_year * 365 + 365), f"Expected {age} < {dod} < {(max_year * 365 + 365)}"

    def test_predicted_day_of_death_0_89(self):
        age = 0
        max_year = 89
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert age < dod < (max_year * 365 + 365), f"Expected {age} < {dod} < {(max_year * 365 + 365)}"

    def test_predicted_day_of_death_30_default(self):
        age = 30 * 365 + 180  # ~30 1/2 years
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age)
            assert age < dod < 36865, f"Expected {age} < {dod} < 36,865"

    def test_predicted_day_of_death_30_100(self):
        age = 30 * 365 + 180  # ~30 1/2 years
        max_year = 100
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert age < dod < (max_year * 365 + 365), f"Expected {age} < {dod} < {(max_year * 365 + 365)}"

    def test_predicted_day_of_death_30_89(self):
        age = 30 * 365 + 180  # ~30 1/2 years
        max_year = 89
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert age < dod < (max_year * 365 + 365), f"Expected {age} < {dod} < {(max_year * 365 + 365)}"

    def test_predicted_day_of_death_default_maximum(self):
        age = 100 * 365  # 100 years
        for _ in range(1024):
            dod = predicted_day_of_death(age)
            assert dod // 365 == 100, f"Expected {dod} // 365 == 100"
            assert (dod - age) < 365, f"Expected {dod} - {age} ({dod-age}) < 365"

    def test_predicted_day_of_death_30_maximum(self):
        max_year = 30
        age = max_year * 365 + 180  # ~30 1/2 years
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert dod // 365 == max_year, f"Expected {dod} // 365 == {max_year}"
            assert (dod - age) < 365, f"Expected {dod} - {age} ({dod-age}) < 365"

    def test_doy_distribution(self):
        for _ in range(1024):
            age_days = np.random.randint(0, 365 * 100)
            dod = predicted_day_of_death(age_days)

            age_years = age_days // 365
            yod = dod // 365
            if yod == age_years:
                # if same year, any day after today is possible
                assert dod % 365 in range(
                    age_days % 365 + 1, 365
                ), f"Expected {dod} % 365 in range({age_days % 365 + 1}, 365) ({age_days=})"
            else:
                # if future year, any day is possible
                assert dod % 365 in range(365), f"Expected {dod} % 365 in range(0, 365) ({age_days=})"

    def test_predicted_day_of_death_30plus363_30max(self):
        # If the agent is Y years and 363 days old, and predicted _year_ of death is Y,
        # then the predicted _day_ of death is December 31th.
        for max_year in [5, 10, 15, 20, 25, 30]:
            age = max_year * 365 + 363  # "December 30th", so to speak
            dod = predicted_day_of_death(age, max_year)
            assert dod // 365 == max_year, f"Expected {dod} // 365 == {max_year}"
            assert dod % 365 == 364, f"Expected {dod} % 365 == 364"  # "December 31st", so to speak

    def test_predicted_day_of_death_30plus364_30max(self):
        # If the agent is Y years and 364 days old, and predicted _year_ of death is Y,
        # then the predicted _day_ of death is January 1st of year Y+1.
        for max_year in [5, 10, 15, 20, 25, 30]:
            age = max_year * 365 + 364  # "December 31st", so to speak
            dod = predicted_day_of_death(age, max_year)
            assert dod // 365 == max_year + 1, f"Expected {dod} // 365 == {max_year + 1}"
            assert dod % 365 == 0, f"Expected {dod} % 365 == 364"  # "January 1st", so to speak


class TestGrabBag(unittest.TestCase):
    """Tests for the GrabBag class."""

    # Test empty grabbag
    def test_empty_grab_bag(self):
        # assert that the grab bag is empty
        gb = GrabBag()
        assert len(gb) == 0

    # Test initialization from a single dictionary
    def test_single_dict_grab_bag(self):
        # assert that the grab bag is initialized with a single dictionary
        gb = GrabBag({"a": 1, "b": 2})
        assert gb.a == 1
        assert gb.b == 2

    # Test inititalization from a single GrabBag
    def test_single_grab_bag(self):
        # assert that the grab bag is initialized with a single GrabBag
        gb = GrabBag(GrabBag({"a": 1, "b": 2}))
        assert gb.a == 1
        assert gb.b == 2

    # Test initialization from multiple dictionaries
    def test_multiple_dict_grab_bag(self):
        # assert that the grab bag is initialized with multiple dictionaries
        gb = GrabBag({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    # Test initialization from multiple GrabBags
    def test_multiple_grab_bag(self):
        # assert that the grab bag is initialized with multiple GrabBags
        gb = GrabBag(GrabBag({"a": 1, "b": 2}), GrabBag({"c": 3, "d": 4}))
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    # Test initialization from a mix of dictionaries and GrabBags
    def test_mixed_grab_bag(self):
        # assert that the grab bag is initialized with a mix of dictionaries and GrabBags
        gb = GrabBag({"a": 1, "b": 2}, GrabBag({"c": 3, "d": 4}))
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    # Test adding a dictionary to an empty grab bag
    def test_add_dict_empty_grab_bag(self):
        # assert that a dictionary can be added to an empty grab bag
        gb = GrabBag()
        gb += {"a": 1, "b": 2}
        assert gb.a == 1
        assert gb.b == 2

    # Test adding a GrabBag to an existing GrabBag
    def test_add_grab_bag(self):
        # assert that a GrabBag can be added to an existing GrabBag
        gb = GrabBag({"a": 1, "b": 2})
        gb += GrabBag({"c": 3, "d": 4})
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    # Test that adding a subsequent dictionary to a GrabBag overrides existing values
    def test_add_dict_override(self):
        # assert that adding a subsequent dictionary to a GrabBag overrides existing values
        gb = GrabBag({"a": 1, "b": 2})
        gb += {"b": 3, "c": 4}
        assert gb.a == 1
        assert gb.b == 3
        assert gb.c == 4

    # Test that adding a subsequent GrabBag to a GrabBag overrides existing values
    def test_add_grab_bag_override(self):
        # assert that adding a subsequent GrabBag to a GrabBag overrides existing values
        gb = GrabBag({"a": 1, "b": 2})
        gb += GrabBag({"b": 3, "c": 4})
        assert gb.a == 1
        assert gb.b == 3
        assert gb.c == 4

    # Test that GrabBag + GrabBag creates a new grab bag _and_ does not alter the existing grab bags
    def test_add_grab_bag_new(self):
        # assert that GrabBag + GrabBag creates a new grab bag _and_ does not alter the existing grab bags
        gb1 = GrabBag({"a": 1, "b": 2})
        gb2 = GrabBag({"b": 3, "c": 4})
        gb3 = gb1 + gb2
        assert gb1.a == 1
        assert gb1.b == 2
        assert gb2.b == 3
        assert gb2.c == 4
        assert gb3.a == 1
        assert gb3.b == 3
        assert gb3.c == 4

    # Test the __str__ method of the GrabBag class
    def test_str(self):
        # assert that the __str__ method returns the expected string
        gb = GrabBag({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert str(gb) == str({"a": 1, "b": 2, "c": 3, "d": 4})

    # Test the __repr__ method of the GrabBag class
    def test_repr(self):
        # assert that the __repr__ method returns the expected string
        gb = GrabBag({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert repr(gb) == f"GrabBag({ {'a': 1, 'b': 2, 'c': 3, 'd': 4}!s})"


if __name__ == "__main__":
    unittest.main()
