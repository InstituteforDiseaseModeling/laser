import unittest

import numpy as np

from idmlaser.numpynumba.demographics import DemographicsByYear
from idmlaser.numpynumba.demographics import DemographicsStatic


class TestDemographicsStatic(unittest.TestCase):
    def setUp(self):
        self.demographics = DemographicsStatic(nyears=5, nnodes=3)
        self.population = np.array([[100, 200, 300], [110, 210, 310], [120, 220, 320], [130, 230, 330], [140, 240, 340]])
        self.births = np.array([[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33], [14, 24, 34]])
        self.deaths = np.array([[5, 10, 15], [6, 11, 16], [7, 12, 17], [8, 13, 18], [9, 14, 19]])
        self.immigrations = np.array([[2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9], [6, 8, 10]])

    def test_initialize(self):
        self.demographics.initialize(
            population=self.population,
            births=self.births,
            deaths=self.deaths,
            immigrations=self.immigrations,
        )
        assert np.array_equal(self.demographics.population, self.population)
        assert np.array_equal(self.demographics.births, self.births)
        assert np.array_equal(self.demographics.deaths, self.deaths)
        assert np.array_equal(self.demographics.immigrations, self.immigrations)

    def test_nyears(self):
        assert self.demographics.nyears == 5

    def test_nnodes(self):
        assert self.demographics.nnodes == 3

    def test_population(self):
        self.demographics.initialize(
            population=self.population,
            births=self.births,
            deaths=self.deaths,
            immigrations=self.immigrations,
        )
        assert np.array_equal(self.demographics.population, self.population)

    def test_births(self):
        self.demographics.initialize(
            population=self.population,
            births=self.births,
            deaths=self.deaths,
            immigrations=self.immigrations,
        )
        assert np.array_equal(self.demographics.births, self.births)

    def test_deaths(self):
        self.demographics.initialize(
            population=self.population,
            births=self.births,
            deaths=self.deaths,
            immigrations=self.immigrations,
        )
        assert np.array_equal(self.demographics.deaths, self.deaths)

    def test_immigrations(self):
        self.demographics.initialize(
            population=self.population,
            births=self.births,
            deaths=self.deaths,
            immigrations=self.immigrations,
        )
        assert np.array_equal(self.demographics.immigrations, self.immigrations)


class TestDemographicsByYear(unittest.TestCase):
    def setUp(self):
        self.demographics = DemographicsByYear(nyears=5, nnodes=3)
        self.initial_population = np.array([100, 200, 300])
        self.cbr = np.array([10, 20, 30, 40, 50])
        self.mortality = np.array([5, 10, 15, 20, 25])
        self.immigration = np.array([2, 4, 6, 8, 10])

    def test_initialize(self):
        self.demographics.initialize(
            initial_population=self.initial_population,
            cbr=self.cbr,
            mortality=self.mortality,
            immigration=self.immigration,
        )
        assert np.array_equal(self.demographics.population[0], self.initial_population)
        assert np.array_equal(self.demographics._cbr, self.cbr)
        assert np.array_equal(self.demographics._mortality, self.mortality)
        assert np.array_equal(self.demographics._immigration, self.immigration)

    def test_population(self):
        self.demographics.initialize(
            initial_population=self.initial_population,
            cbr=self.cbr,
            mortality=self.mortality,
            immigration=self.immigration,
        )
        assert np.array_equal(self.demographics.population[0], self.initial_population)

    def test_constant_cbr(self):
        self.demographics.initialize(
            initial_population=self.initial_population,
            cbr=self.cbr[0],
            mortality=0,
            immigration=0,
        )
        assert np.array_equal(self.demographics.population[0], self.initial_population)
        assert np.array_equal(self.demographics.births, np.round(self.demographics.population * self.cbr[0] / 1000.0))
        assert np.array_equal(self.demographics.population[1:], np.round(self.demographics.population[:-1] * (1 + (self.cbr[0] / 1000.0))))

    def test_constant_mortality(self):
        self.demographics.initialize(
            initial_population=self.initial_population,
            cbr=0,
            mortality=self.mortality[0],
            immigration=0,
        )
        assert np.array_equal(self.demographics.population[0], self.initial_population)
        assert np.array_equal(self.demographics.deaths, np.round(self.demographics.population * self.mortality[0] / 1000.0))
        assert np.array_equal(
            self.demographics.population[1:], np.round(self.demographics.population[:-1] * (1 - (self.mortality[0] / 1000.0)))
        )

    def test_constant_immigration(self):
        self.demographics.initialize(
            initial_population=self.initial_population,
            cbr=0,
            mortality=0,
            immigration=self.immigration[0],
        )
        assert np.array_equal(self.demographics.population[0], self.initial_population)
        assert np.array_equal(self.demographics.immigrations, np.round(self.demographics.population * self.immigration[0] / 1000.0))
        assert np.array_equal(
            self.demographics.population[1:], np.round(self.demographics.population[:-1] * (1 + (self.immigration[0] / 1000.0)))
        )

    def test_births(self):
        self.demographics.initialize(
            initial_population=self.initial_population,
            cbr=self.cbr,
            mortality=self.mortality,
            immigration=self.immigration,
        )
        assert (self.demographics.births[0] == np.round(self.initial_population * self.cbr[0] / 1000.0)).all()

    def test_deaths(self):
        self.demographics.initialize(
            initial_population=self.initial_population,
            cbr=self.cbr,
            mortality=self.mortality,
            immigration=self.immigration,
        )
        assert (self.demographics.deaths[0] == np.round(self.initial_population * self.mortality[0] / 1000.0)).all()

    def test_immigrations(self):
        self.demographics.initialize(
            initial_population=self.initial_population,
            cbr=self.cbr,
            mortality=self.mortality,
            immigration=self.immigration,
        )
        assert (self.demographics.immigrations[0] == np.round(self.initial_population * self.immigration[0] / 1000.0)).all()


if __name__ == "__main__":
    unittest.main()
