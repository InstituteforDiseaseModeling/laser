import csv
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

from laser_core.migration import distance
from laser_core.utils import calc_capacity
from laser_core.utils import calc_distances

City = namedtuple("City", ["name", "pop", "lat", "long"])


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


if __name__ == "__main__":
    unittest.main()
