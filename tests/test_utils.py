"""Tests for the utils module."""

import unittest

import numpy as np
import pytest

from idmlaser.utils import PriorityQueue
from idmlaser.utils import PriorityQueueNB
from idmlaser.utils import PriorityQueueNP
from idmlaser.utils import PropertySet
from idmlaser.utils import daily_births_deaths_from_annual
from idmlaser.utils import pop_from_cbr_and_mortality
from idmlaser.utils import predicted_day_of_death
from idmlaser.utils import predicted_year_of_death


class TestPriorityQueue(unittest.TestCase):
    """Tests for the PriorityQueue class."""

    def setUp(self):
        self.pq = PriorityQueue(3)

    def test_push_pop(self):
        """Test pushing and popping elements from the priority queue."""
        self.pq.push(1, 2)
        self.pq.push(2, 1)
        self.pq.push(3, 3)

        assert self.pq.pop() == (2, 1)
        assert self.pq.pop() == (1, 2)
        assert self.pq.pop() == (3, 3)

    def test_push_pop_random(self):
        """Test pushing and popping random elements from the priority queue."""
        values = np.random.randint(0, 100, 1024, dtype=np.uint32)
        self.pq = PriorityQueue(len(values))
        for i in values:
            self.pq.push(i, i)
        minimum = 0
        while len(self.pq) > 0:
            value, _ = self.pq.pop()
            assert value >= minimum
            minimum = value

    def test_peek(self):
        """Test peeking at the top element of the priority queue."""
        self.pq.push(1, 2)
        self.pq.push(2, 1)
        self.pq.push(3, 3)

        assert self.pq.peek() == (2, 1)

    def test_empty_peek(self):
        """Test peeking at the top element of an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.pq.peek()

    def test_peek_random(self):
        """Test peeking at the top element of the priority queue with random values."""
        values = np.random.randint(0, 100, 1024, dtype=np.uint32)
        self.pq = PriorityQueue(len(values))
        for i in values:
            self.pq.push(i, i)
        minimum = values.min()
        assert self.pq.peek() == (minimum, minimum)

    def test_empty_pop(self):
        """Test popping from an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            self.pq.pop()

    def test_full_push(self):
        """Test pushing to a full priority queue. Should raise an IndexError."""
        self.pq.push(1, 2)
        self.pq.push(2, 1)
        self.pq.push(3, 3)

        with pytest.raises(IndexError):
            self.pq.push(4, 4)

    def test_push_timing(self):
        """Test the timing of the push method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 20
        values = np.random.randint(0, 100, count, dtype=np.int32)
        self.pq = PriorityQueue(len(values))
        elapsed = timeit.timeit(
            "for i, value in enumerate(values): self.pq.push(i, value)", globals={"values": values, "self": self}, number=1
        )
        print(
            f"\n PriorityQueue.push() timing:   {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )

        # print(timeit.timeit("self.pq.pop()", globals=globals(), number=1024))

    def test_pop_timing(self):
        """Test the timing of the pop method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 16
        values = np.random.randint(0, 100, count, dtype=np.int32)
        self.pq = PriorityQueue(len(values))
        for i, value in enumerate(values):
            self.pq.push(i, value)

        elapsed = timeit.timeit("while len(self.pq): self.pq.pop()", globals={"self": self}, number=1)
        print(
            f"\n PriorityQueue.pop() timing:    {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )


class TestPopFromCbrAndMortality(unittest.TestCase):
    """Tests for the pop_from_cbr_and_mortality function."""

    INITIAL = 100_000
    CBR = 25  # 25/1000 = 2.5% growth per annum
    MORTALITY = 17  # 17/1000 = 1.7% mortality per annum ~60 years life expectancy
    NYEARS = 50

    def test_pop_from_cbr_constant_and_mortality_constant(self):
        """Test population growth from constant CBR and constant mortality. Stochastic results so run multiple times."""
        failed = 0
        for _ in range(10):
            births, deaths, population = pop_from_cbr_and_mortality(self.INITIAL, self.CBR, self.MORTALITY, self.NYEARS)

            assert len(births) == self.NYEARS
            assert len(deaths) == self.NYEARS
            assert len(population) == (self.NYEARS + 1)

            for i in range(len(population) - 1):
                assert population[i + 1] == population[i] + births[i] - deaths[i]

            expected_population = np.uint32(np.round(self.INITIAL * (1 + (self.CBR - self.MORTALITY) / 1000) ** self.NYEARS))
            if abs(1 - population[-1] / expected_population) >= 0.01:
                print(
                    f"\nExpected {expected_population}, got {population[-1]} ({100 * abs(1 - population[-1] / expected_population)}% difference)"
                )
                failed += 1

        assert failed <= 1, f"Failed {failed} out of 10 runs"

        return

    def test_pop_from_cbr_varying_and_mortality_constant(self):
        """Test population growth from varying CBR and constant mortality. Stochastic results so run multiple times."""
        failed = 0
        for _ in range(10):
            cbr = np.random.normal(self.CBR, 3, self.NYEARS)

            births, deaths, population = pop_from_cbr_and_mortality(self.INITIAL, cbr, self.MORTALITY, self.NYEARS)

            assert len(births) == self.NYEARS
            assert len(deaths) == self.NYEARS
            assert len(population) == (self.NYEARS + 1)

            for i in range(len(population) - 1):
                assert population[i + 1] == population[i] + births[i] - deaths[i]

            expected_population = np.uint32(np.round(self.INITIAL * (1 + (cbr.mean() - self.MORTALITY) / 1000) ** self.NYEARS))
            if abs(1 - population[-1] / expected_population) >= 0.01:
                print(
                    f"\nExpected {expected_population}, got {population[-1]} ({100 * abs(1 - population[-1] / expected_population)}% difference)"
                )
                failed += 1

        assert failed <= 1, f"Failed {failed} out of 10 runs"

        return

    def test_pop_from_cbr_constant_and_mortality_varying(self):
        """Test population growth from constant CBR and varying mortality. Stochastic results so run multiple times."""
        failed = 0
        for _ in range(10):
            mortality = np.random.normal(self.MORTALITY, 2, self.NYEARS)

            births, deaths, population = pop_from_cbr_and_mortality(self.INITIAL, self.CBR, mortality, self.NYEARS)

            assert len(births) == self.NYEARS
            assert len(deaths) == self.NYEARS
            assert len(population) == (self.NYEARS + 1)

            for i in range(len(population) - 1):
                assert population[i + 1] == population[i] + births[i] - deaths[i]

            expected_population = np.uint32(np.round(self.INITIAL * (1 + (self.CBR - mortality.mean()) / 1000) ** self.NYEARS))
            if abs(1 - population[-1] / expected_population) >= 0.01:
                print(
                    f"\nExpected {expected_population}, got {population[-1]} ({100 * abs(1 - population[-1] / expected_population)}% difference)"
                )
                failed += 1

        assert failed <= 1, f"Failed {failed} out of 10 runs"

        return

    def test_pop_from_cbr_varying_and_mortality_varying(self):
        """Test population growth from varying CBR and varying mortality. Stochastic results so run multiple times."""
        failed = 0
        for _ in range(10):
            cbr = np.random.normal(self.CBR, 3, self.NYEARS)
            mortality = np.random.normal(self.MORTALITY, 2, self.NYEARS)

            births, deaths, population = pop_from_cbr_and_mortality(self.INITIAL, cbr, mortality, self.NYEARS)

            assert len(births) == self.NYEARS
            assert len(deaths) == self.NYEARS
            assert len(population) == (self.NYEARS + 1)

            for i in range(len(population) - 1):
                assert population[i + 1] == population[i] + births[i] - deaths[i]

            expected_population = np.uint32(np.round(self.INITIAL * (1 + (cbr.mean() - mortality.mean()) / 1000) ** self.NYEARS))
            if abs(1 - population[-1] / expected_population) >= 0.01:
                print(
                    f"\nExpected {expected_population}, got {population[-1]} ({100 * abs(1 - population[-1] / expected_population)}% difference)"
                )
                failed += 1

        assert failed <= 1, f"Failed {failed} out of 10 runs"

        return


class TestDailyBirthsDeathsFromAnnual(unittest.TestCase):
    """Tests for the daily_births_deaths_from_annual function."""

    NYEARS = 10

    def test_from_fixed_values(self):
        """Test daily births and deaths from fixed annual values."""
        annual_births = np.full(self.NYEARS, 730)  # 2 births per day
        annual_deaths = np.full(self.NYEARS, 365)  # 1 death per day

        daily_births, daily_deaths = daily_births_deaths_from_annual(annual_births, annual_deaths)

        # Assert that the actual daily births and deaths match the expected values
        assert np.all(daily_births == 2), f"Expected 2, got {daily_births}"
        assert np.all(daily_deaths == 1), f"Expected 1, got {daily_deaths}"
        assert daily_births.sum() == (self.NYEARS * 730), f"Expected {self.NYEARS * 730}, got {daily_births.sum()}"
        assert daily_deaths.sum() == (self.NYEARS * 365), f"Expected {self.NYEARS * 365}, got {daily_deaths.sum()}"

    def test_from_stochastic_values(self):
        """Test daily births and deaths from stochastic annual values."""
        annual_births = np.random.poisson(730, self.NYEARS)  # 2 births per day or so
        annual_deaths = np.random.poisson(365, self.NYEARS)  # 1 death per day or so

        daily_births, daily_deaths = daily_births_deaths_from_annual(annual_births, annual_deaths)

        # Assert that the actual daily births and deaths match the expected values
        assert np.all(daily_births >= 0), f"Expected >= 0, got {daily_births}"
        assert np.all(daily_deaths >= 0), f"Expected >= 0, got {daily_deaths}"
        assert daily_births.sum() == annual_births.sum(), f"Expected {annual_births.sum()}, got {daily_births.sum()}"
        assert daily_deaths.sum() == annual_deaths.sum(), f"Expected {annual_deaths.sum()}, got {daily_deaths.sum()}"
        self.assertAlmostEqual(daily_births.mean(), annual_births.mean() / 365, places=4)  # noqa: PT009
        self.assertAlmostEqual(daily_deaths.mean(), annual_deaths.mean() / 365, places=4)  # noqa: PT009


class TestPredictedDateOfDeath(unittest.TestCase):
    """Tests for the predicted_year_of_death and predicted_day_of_death functions."""

    def test_predicted_year_of_death_0_default(self):
        """Test predicted year of death for an agent aged 0 with default maximum year."""
        age = 0
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age)
            assert age <= yod <= 100, f"Expected {age} <= {yod} <= 100"

    def test_predicted_year_of_death_0_100(self):
        """Test predicted year of death for an agent aged 0 with a maximum year of 100."""
        age = 0
        max_year = 100
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert age <= yod <= max_year, f"Expected {age} <= {yod} <= {max_year}"

    def test_predicted_year_of_death_0_89(self):
        """Test predicted year of death for an agent aged 0 with a maximum year of 89."""
        age = 0
        max_year = 89
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert age <= yod <= max_year, f"Expected {age} <= {yod} <= {max_year}"

    def test_predicted_year_of_death_30_default(self):
        """Test predicted year of death for an agent aged 30 with default maximum year."""
        age = 30
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age)
            assert age <= yod <= 100, f"Expected {age} <= {yod} <= 100"

    def test_predicted_year_of_death_30_100(self):
        """Test predicted year of death for an agent aged 30 with a maximum year of 100."""
        age = 30
        max_year = 100
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert age <= yod <= max_year, f"Expected {age} <= {yod} <= {max_year}"

    def test_predicted_year_of_death_30_89(self):
        """Test predicted year of death for an agent aged 30 with a maximum year of 89."""
        age = 30
        max_year = 89
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert age <= yod <= max_year, f"Expected {age} <= {yod} <= {max_year}"

    def test_predicted_year_of_death_default_maximum(self):
        """Test predicted year of death for an agent aged 100 with default maximum year."""
        age = 100
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age)
            assert yod == 100, f"Expected {yod} == 100"

    def test_predicted_year_of_death_30_maximum(self):
        """Test predicted year of death for an agent aged 30 with a maximum year of 30."""
        age = max_year = 30
        for _ in range(1024):  # 1000-ish trials
            yod = predicted_year_of_death(age, max_year)
            assert yod == max_year, f"Expected {yod} == {max_year}"

    def test_predicted_day_of_death_0_default(self):
        """Test predicted day of death for an agent aged 0 with default maximum year."""
        age = 0
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age)
            assert age < dod < 36865, f"Expected {age} < {dod} < 36,865 (365 * 100 + 365)"

    def test_predicted_day_of_death_0_100(self):
        """Test predicted day of death for an agent aged 0 with a maximum year of 100."""
        age = 0
        max_year = 100
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert age < dod < (max_year * 365 + 365), f"Expected {age} < {dod} < {(max_year * 365 + 365)}"

    def test_predicted_day_of_death_0_89(self):
        """Test predicted day of death for an agent aged 0 with a maximum year of 89."""
        age = 0
        max_year = 89
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert age < dod < (max_year * 365 + 365), f"Expected {age} < {dod} < {(max_year * 365 + 365)}"

    def test_predicted_day_of_death_30_default(self):
        """Test predicted day of death for an agent aged 30.5 with default maximum year."""
        age = 30 * 365 + 180  # ~30 1/2 years
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age)
            assert age < dod < 36865, f"Expected {age} < {dod} < 36,865"

    def test_predicted_day_of_death_30_100(self):
        """Test predicted day of death for an agent aged 30.5 with a maximum year of 100."""
        age = 30 * 365 + 180  # ~30 1/2 years
        max_year = 100
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert age < dod < (max_year * 365 + 365), f"Expected {age} < {dod} < {(max_year * 365 + 365)}"

    def test_predicted_day_of_death_30_89(self):
        """Test predicted day of death for an agent aged 30.5 with a maximum year of 89."""
        age = 30 * 365 + 180  # ~30 1/2 years
        max_year = 89
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert age < dod < (max_year * 365 + 365), f"Expected {age} < {dod} < {(max_year * 365 + 365)}"

    def test_predicted_day_of_death_default_maximum(self):
        """Test predicted day of death for an agent aged 100 with default maximum year."""
        age = 100 * 365  # 100 years
        for _ in range(1024):
            dod = predicted_day_of_death(age)
            assert dod // 365 == 100, f"Expected {dod} // 365 == 100"
            assert (dod - age) < 365, f"Expected {dod} - {age} ({dod-age}) < 365"

    def test_predicted_day_of_death_30_maximum(self):
        """Test predicted day of death for an agent aged 30 with a maximum year of 30."""
        max_year = 30
        age = max_year * 365 + 180  # ~30 1/2 years
        for _ in range(1024):  # 1000-ish trials
            dod = predicted_day_of_death(age, max_year)
            assert dod // 365 == max_year, f"Expected {dod} // 365 == {max_year}"
            assert (dod - age) < 365, f"Expected {dod} - {age} ({dod-age}) < 365"

    def test_doy_distribution(self):
        """Test the distribution of the day of the year for the predicted day of death."""
        for _ in range(1024):
            age_days = np.random.randint(0, 365 * 100, dtype=np.uint16)
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
        """Test predicted day of death for an agent aged 30 years and 363 days with a maximum year of 30."""
        # If the agent is Y years and 363 days old, and predicted _year_ of death is Y,
        # then the predicted _day_ of death is December 31th.
        for max_year in [5, 10, 15, 20, 25, 30]:
            age = max_year * 365 + 363  # "December 30th", so to speak
            dod = predicted_day_of_death(age, max_year)
            assert dod // 365 == max_year, f"Expected {dod} // 365 == {max_year}"
            assert dod % 365 == 364, f"Expected {dod} % 365 == 364"  # "December 31st", so to speak

    def test_predicted_day_of_death_30plus364_30max(self):
        """Test predicted day of death for an agent aged 30 years and 364 days with a maximum year of 30."""
        # If the agent is Y years and 364 days old, and predicted _year_ of death is Y,
        # then the predicted _day_ of death is January 1st of year Y+1.
        for max_year in [5, 10, 15, 20, 25, 30]:
            age = max_year * 365 + 364  # "December 31st", so to speak
            dod = predicted_day_of_death(age, max_year)
            assert dod // 365 == max_year + 1, f"Expected {dod} // 365 == {max_year + 1}"
            assert dod % 365 == 0, f"Expected {dod} % 365 == 364"  # "January 1st", so to speak


class TestPropertySet(unittest.TestCase):
    """Tests for the PropertySet class."""

    def test_empty_property_set(self):
        """Test an empty PropertySet."""
        # assert that the grab bag is empty
        gb = PropertySet()
        assert len(gb) == 0

    def test_single_dict_property_set(self):
        """Test initialization from a single dictionary."""
        # assert that the grab bag is initialized with a single dictionary
        gb = PropertySet({"a": 1, "b": 2})
        assert gb.a == 1
        assert gb.b == 2

    def test_single_property_set(self):
        """Test initialization from a single PropertySet."""
        # assert that the grab bag is initialized with a single PropertySet
        gb = PropertySet(PropertySet({"a": 1, "b": 2}))
        assert gb.a == 1
        assert gb.b == 2

    def test_multiple_dict_property_set(self):
        """Test initialization from multiple dictionaries."""
        # assert that the grab bag is initialized with multiple dictionaries
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    def test_multiple_property_set(self):
        """Test initialization from multiple PropertySets."""
        # assert that the grab bag is initialized with multiple PropertySets
        gb = PropertySet(PropertySet({"a": 1, "b": 2}), PropertySet({"c": 3, "d": 4}))
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    def test_mixed_property_set(self):
        """Test initialization from a mix of dictionaries and PropertySets."""
        # assert that the grab bag is initialized with a mix of dictionaries and PropertySets
        gb = PropertySet({"a": 1, "b": 2}, PropertySet({"c": 3, "d": 4}))
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    def test_add_dict_empty_property_set(self):
        """Test adding a dictionary to an empty PropertySet."""
        # assert that a dictionary can be added to an empty grab bag
        gb = PropertySet()
        gb += {"a": 1, "b": 2}
        assert gb.a == 1
        assert gb.b == 2

    def test_add_property_set(self):
        """Test adding a PropertySet to an existing PropertySet."""
        # assert that a PropertySet can be added to an existing PropertySet
        gb = PropertySet({"a": 1, "b": 2})
        gb += PropertySet({"c": 3, "d": 4})
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    def test_add_dict_override(self):
        """Test that adding a subsequent dictionary to a PropertySet overrides existing values."""
        # assert that adding a subsequent dictionary to a PropertySet overrides existing values
        gb = PropertySet({"a": 1, "b": 2})
        gb += {"b": 3, "c": 4}
        assert gb.a == 1
        assert gb.b == 3
        assert gb.c == 4

    def test_add_property_set_override(self):
        """Test that adding a subsequent PropertySet to a PropertySet overrides existing values."""
        # assert that adding a subsequent PropertySet to a PropertySet overrides existing values
        gb = PropertySet({"a": 1, "b": 2})
        gb += PropertySet({"b": 3, "c": 4})
        assert gb.a == 1
        assert gb.b == 3
        assert gb.c == 4

    def test_add_property_set_new(self):
        """Test that PropertySet + PropertySet creates a new PropertySet _and_ does not alter the existing PropertySets"""
        # assert that PropertySet + PropertySet creates a new grab bag _and_ does not alter the existing grab bags
        gb1 = PropertySet({"a": 1, "b": 2})
        gb2 = PropertySet({"b": 3, "c": 4})
        gb3 = gb1 + gb2
        assert gb1.a == 1
        assert gb1.b == 2
        assert gb2.b == 3
        assert gb2.c == 4
        assert gb3.a == 1
        assert gb3.b == 3
        assert gb3.c == 4

    def test_str(self):
        """Test the __str__ method of the PropertySet class."""
        # assert that the __str__ method returns the expected string
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert str(gb) == str({"a": 1, "b": 2, "c": 3, "d": 4})

    def test_repr(self):
        """Test the __repr__ method of the PropertySet class."""
        # assert that the __repr__ method returns the expected string
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert repr(gb) == f"PropertySet({ {'a': 1, 'b': 2, 'c': 3, 'd': 4}!s})"


class TestPriorityQueueNP(unittest.TestCase):
    """Tests for the PriorityQueueNP class."""

    def setUp(self):
        # 31 41 59 26 53 58 97
        self.pq = PriorityQueueNP(7, np.array([31, 41, 59, 26, 53, 58, 97]))

    def test_push_pop(self):
        """Test pushing and popping elements from the priority queue."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        assert self.pq.popiv() == (3, 26)
        assert self.pq.popiv() == (0, 31)
        assert self.pq.popiv() == (1, 41)
        assert self.pq.popiv() == (4, 53)
        assert self.pq.popiv() == (5, 58)
        assert self.pq.popiv() == (2, 59)
        assert self.pq.popiv() == (6, 97)

    def test_push_pop_random(self):
        """Test pushing and popping random values from the priority queue."""
        values = np.random.randint(0, 100, 1024, dtype=np.uint32)
        self.pq = PriorityQueueNP(len(values), values)
        for i in range(len(values)):
            self.pq.push(i)
        minimum = 0
        while len(self.pq) > 0:
            value = self.pq.popv()
            assert value >= minimum
            minimum = value

    def test_peek(self):
        """Test peeking at the top element of the priority queue."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        assert self.pq.peekiv() == (3, 26)

    def test_empty_peek(self):
        """Test peeking at the top element of an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.pq.peekiv()

    def test_peek_random(self):
        """Test peeking at the top element of the priority queue with random values."""
        values = np.random.randint(0, 100, 1024, dtype=np.uint32)
        self.pq = PriorityQueueNP(len(values), values)
        for i in range(len(values)):
            self.pq.push(i)
        minimum = values.min()
        assert self.pq.peekv() == minimum

    def test_empty_pop(self):
        """Test popping from an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            self.pq.pop()

    def test_full_push(self):
        """Test pushing to a full priority queue. Should raise an IndexError."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        with pytest.raises(IndexError):
            self.pq.push(7)

    def test_push_timing(self):
        """Test the timing of the push method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 20
        values = np.random.randint(0, 100, count, dtype=np.uint32)
        self.pq = PriorityQueueNP(len(values), values)
        self.pq.push(0)  # in case we need to compile the push method
        self.pq.pop()  # in case we need to compile the pop method
        elapsed = timeit.timeit("for i in range(len(values)): self.pq.push(i)", globals={"values": values, "self": self}, number=1)
        print(
            f"\n PriorityQueueNP.push() timing: {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )

    def test_pop_timing(self):
        """Test the timing of the pop method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 16
        values = np.random.randint(0, 100, count, dtype=np.uint32)
        self.pq = PriorityQueueNP(len(values), values)
        self.pq.push(0)  # in case we need to compile the push method
        self.pq.pop()  # in case we need to compile the pop method
        for i in range(len(values)):
            self.pq.push(i)

        elapsed = timeit.timeit("while len(self.pq): self.pq.popv()", globals={"self": self}, number=1)
        print(
            f"\n PriorityQueueNP.popv() timing: {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )


class TestPriorityQueueNB(unittest.TestCase):
    """Tests for the PriorityQueueNB class."""

    def setUp(self):
        # 31 41 59 26 53 58 97
        self.pq = PriorityQueueNB(7, np.array([31, 41, 59, 26, 53, 58, 97], dtype=np.int32))

    def test_push_pop(self):
        """Test pushing and popping elements from the priority queue."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        assert self.pq.popiv() == (3, 26)
        assert self.pq.popiv() == (0, 31)
        assert self.pq.popiv() == (1, 41)
        assert self.pq.popiv() == (4, 53)
        assert self.pq.popiv() == (5, 58)
        assert self.pq.popiv() == (2, 59)
        assert self.pq.popiv() == (6, 97)

    def test_push_pop_random(self):
        """Test pushing and popping random values from the priority queue."""
        values = np.random.randint(0, 100, 1024, dtype=np.int32)
        self.pq = PriorityQueueNB(len(values), values)
        for i in range(len(values)):
            self.pq.push(i)
        minimum = 0
        while len(self.pq) > 0:
            value = self.pq.popv()
            assert value >= minimum
            minimum = value

    def test_peek(self):
        """Test peeking at the top element of the priority queue."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        assert self.pq.peekiv() == (3, 26)

    def test_empty_peek(self):
        """Test peeking at the top element of an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.pq.peekiv()

    def test_peek_random(self):
        """Test peeking at the top element of the priority queue with random values."""
        values = np.random.randint(0, 100, 1024, dtype=np.int32)
        self.pq = PriorityQueueNB(len(values), values)
        for i in range(len(values)):
            self.pq.push(i)
        minimum = values.min()
        assert self.pq.peekv() == minimum

    def test_empty_pop(self):
        """Test popping from an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            self.pq.pop()

    def test_full_push(self):
        """Test pushing to a full priority queue. Should raise an IndexError."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        with pytest.raises(IndexError):
            self.pq.push(7)

    def test_push_timing(self):
        """Test the timing of the push method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 20
        values = np.random.randint(0, 100, count, dtype=np.int32)
        self.pq = PriorityQueueNB(len(values), values)
        elapsed = timeit.timeit("for i in range(len(values)): self.pq.push(i)", globals={"values": values, "self": self}, number=1)
        print(
            f"\n PriorityQueueNB.push() timing: {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )

        # print(timeit.timeit("self.pq.popv()", globals=globals(), number=1024))

    def test_pop_timing(self):
        """Test the timing of the pop method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 20
        values = np.random.randint(0, 100, count, dtype=np.int32)
        self.pq = PriorityQueueNB(len(values), values)
        for i in range(len(values)):
            self.pq.push(i)

        elapsed = timeit.timeit("while len(self.pq): self.pq.popv()", globals={"self": self}, number=1)
        print(
            f"\n PriorityQueueNB.popv() timing: {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )


if __name__ == "__main__":
    unittest.main()
