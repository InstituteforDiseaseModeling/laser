import csv
import unittest
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np

from laser_core import LaserFrame
from laser_core import PropertySet
from laser_core.migration import distance
from laser_core.utils import calc_capacity
from laser_core.utils import calc_distances
from laser_core.utils import seed_infections_in_patch
from laser_core.utils import seed_infections_randomly

City = namedtuple("City", ["name", "pop", "lat", "long"])


class Model:
    def __init__(self, inf_mean=42, num_agents=1_000_000):
        self.prng = np.random.default_rng(datetime.now().microsecond)  # noqa: DTZ005
        self.agents = LaserFrame(num_agents)
        self.agents.add_scalar_property("nodeid", dtype=np.uint16, default=0)
        self.agents.add_scalar_property("susceptibility", dtype=np.uint8, default=1)
        self.agents.add_scalar_property("itimer", dtype=np.uint8, default=0)
        self.agents.nodeid[:] = (np.arange(self.agents.count) // (num_agents // 100)).astype(self.agents.nodeid.dtype)
        self.params = PropertySet({"inf_mean": inf_mean})


class TestUtilityFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def transmogrify(row):
            name_col = 0
            pop_col = 2
            lat_col = 9
            long_col = 10
            # [:-2] removes the degree symbol and "N" or "W"
            # -long because all US cities are West
            return City(
                name=row[name_col], pop=int(row[pop_col].replace(",", "")), lat=float(row[lat_col][:-2]), long=-float(row[long_col][:-2])
            )

        cities = Path(__file__).parent.absolute() / "data" / "us-cities.csv"
        with cities.open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            cls.header = next(reader)
            cls.city_data = [transmogrify(row) for row in reader]

        cls.top_ten = cls.city_data[0:10]

    def test_calc_distances(self):
        latitudes = np.array([city.lat for city in self.top_ten])
        longitudes = np.array([city.long for city in self.top_ten])

        distances = calc_distances(latitudes, longitudes)

        assert distances.shape == (10, 10), f"Expected shape (10, 10), got {distances.shape}"
        assert distances[0, 0] == 0.0, f"Expected distances[0, 0] == 0.0, got {distances[0, 0]}"
        assert np.all(distances == distances.T), "Expected distances to be symmetric"
        assert distances[0, 1] == np.float32(
            distance(latitudes[0], longitudes[0], latitudes[1], longitudes[1])
        ), f"Expected distance from New York to Los Angeles to be {np.float32(distance(latitudes[0], longitudes[0], latitudes[1], longitudes[1]))}, got {distances[0, 1]}"
        assert distances[0, 2] == np.float32(
            distance(latitudes[0], longitudes[0], latitudes[2], longitudes[2])
        ), f"Expected distance from New York to Chicago to be {np.float32(distance(latitudes[0], longitudes[0], latitudes[2], longitudes[2]))}, got {distances[0, 2]}"
        assert distances[1, 2] == np.float32(
            distance(latitudes[1], longitudes[1], latitudes[2], longitudes[2])
        ), f"Expected distance from Los Angeles to Chicago to be {np.float32(distance(latitudes[1], longitudes[1], latitudes[2], longitudes[2]))}, got {distances[1, 2]}"

        return

    def test_calc_capacity(self):
        population = 1000
        nticks = 5 * 365  # 5 years
        cbr = 20  # 2% annual growth rate

        capacity = calc_capacity(population, nticks, cbr)

        assert capacity == 1105, f"Expected capacity = 1105, got {capacity}"

        return

    def test_seed_infections_randomly_all_naive(self):
        inf_mean = 42
        num_agents = 1_000_000
        model = Model(inf_mean=inf_mean, num_agents=num_agents)
        ninfections = 1024
        seed_infections_randomly(model, ninfections)
        assert (
            np.sum(model.agents.itimer == inf_mean) == ninfections
        ), f"Expected {ninfections} infections, found {np.sum(model.agents.itimer == inf_mean)}"

        return

    def test_seed_infections_randomly_some_immune(self):
        inf_mean = 42
        num_agents = 1_000_000
        model = Model(inf_mean=inf_mean, num_agents=num_agents)
        immune = model.prng.binomial(1, 0.75, size=num_agents)
        model.agents.susceptibility[immune == 1] = 0
        ninfections = 1024
        seed_infections_randomly(model, ninfections)
        assert (
            np.sum(model.agents.itimer == inf_mean) == ninfections
        ), f"Expected {ninfections} infections, found {np.sum(model.agents.itimer == inf_mean)}"

        return

    def test_seed_infections_in_patch_all_naive(self):
        inf_mean = 42
        num_agents = 1_000_000
        model = Model(inf_mean=inf_mean, num_agents=num_agents)
        ninfections = 1024
        ipatch = 13
        seed_infections_in_patch(model, ipatch, ninfections)
        assert (
            np.sum(model.agents.itimer == inf_mean) == ninfections
        ), f"Expected {ninfections} infections, found {np.sum(model.agents.itimer == inf_mean)}"

        return

    def test_seed_infections_in_patch_some_immune(self):
        inf_mean = 42
        num_agents = 1_000_000
        model = Model(inf_mean=inf_mean, num_agents=num_agents)
        immune = model.prng.binomial(1, 0.75, size=num_agents)
        model.agents.susceptibility[immune == 1] = 0
        ninfections = 1024
        ipatch = 13
        seed_infections_in_patch(model, ipatch, ninfections)
        assert (
            np.sum(model.agents.itimer == inf_mean) == ninfections
        ), f"Expected {ninfections} infections, found {np.sum(model.agents.itimer == inf_mean)}"

        return


if __name__ == "__main__":
    unittest.main()
