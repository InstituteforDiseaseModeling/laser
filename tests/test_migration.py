import csv
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

from laser_core.migration import competing_destinations
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import radiation
from laser_core.migration import stouffer

City = namedtuple("City", ["name", "pop", "lat", "long"])


class TestMigrationFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def transmogrify(row):
            name_col = 0
            pop_col = 2
            lat_col = 9
            long_col = 10
            # [:-2] removes the degree symbol and "N" or "W"
            return City(
                name=row[name_col], pop=int(row[pop_col].replace(",", "")), lat=float(row[lat_col][:-2]), long=float(row[long_col][:-2])
            )

        cities = Path(__file__).parent.absolute() / "data" / "us-cities.csv"
        with cities.open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            cls.header = next(reader)
            cls.city_data = [transmogrify(row) for row in reader]

        top_ten = cls.city_data[0:10]
        cls.pops = np.array([city.pop for city in top_ten])
        cls.distances = np.array([[distance(city1.lat, city1.long, city2.lat, city2.long) for city2 in top_ten] for city1 in top_ten])

        return

    def test_setup(self):
        """Test the setup of the test class."""
        assert self.header == [
            "City",
            "State",
            "2023 Estimate",
            "2020 Census",
            "Change",
            "mi^2",
            "km^2",
            "density/mi^2",
            "density/km^2",
            "latitude",
            "longitude",
        ], print(f"us-cities.csv header: {self.header} doesn't match expected header")
        assert len(self.city_data) == 336, print(f"us-cities.csv has {len(self.city_data)} rows, expected 336")
        assert self.city_data[0] == City(name="New York", pop=8258035, lat=40.66, long=73.94), print(
            f"{self.city_data[0].name=} != 'New York'"
        )
        assert self.city_data[-1] == City(name="Davenport", pop=100354, lat=41.56, long=90.60), print(
            f"{self.city_data[-1].name=} != 'Davenport'"
        )

        assert len(self.pops) == 10, print(f"self.pops has {len(self.pops)} elements, expected 10")
        assert self.distances.shape == (10, 10), print(f"self.distances has shape {self.distances.shape}, expected (10, 10)")
        assert np.all(self.distances.diagonal() == 0), print("self.distances.diagonal() != 0")
        assert np.all(self.distances == self.distances.T), print("self.distances is not symmetric")

        assert np.isclose(self.distances[0, 1], 3957.13675, atol=0.00001), print(
            f"New York to Los Angeles distance is {self.distances[0, 1]}, expected 3957.13675"
        )
        assert np.isclose(self.distances[0, 9], 1342.25107, atol=0.00001), print(
            f"New York to Jacksonville distance is {self.distances[0, 9]}, expected 1342.25107"
        )

        return

    def test_gravity_model(self):
        """Test the gravity model migration function without maximum."""
        (k, a, b, c) = (0.1, 0.5, 1.0, 2.0)
        network = gravity(self.pops, self.distances, k=k, a=a, b=b, c=c)
        for _ in range(10):  # check 10 random values
            i = np.random.randint(0, 10)
            j = np.random.randint(0, 10)
            if i != j:
                dist = k * (self.pops[i] ** a) * (self.pops[j] ** b) * self.distances[i, j] ** (-1 * c)
                assert np.isclose(network[i, j], dist), print(f"network[{i}, {j}] = {network[i, j]}, expected {dist=}")
            else:
                assert network[i, j] == 0, print(f"network[{i}, {j}] = {network[i, j]}, expected 0")

        return

    @unittest.skip("Not yet implemented")
    def test_gravity_model_max_frac(self):
        """Test the gravity model migration function with maximum."""
        (k, a, b, c) = (0.1, 0.5, 1.0, 2.0)
        _network = gravity(self.pops, self.distances, k=k, a=a, b=b, c=c, max_frac=0.1)
        # TBD
        raise AssertionError("Test not yet implemented")

    @unittest.skip("Not yet implemented")
    def test_competing_destinations(self):
        """Test the competing destinations migration function."""
        (b, c, delta) = (1.0, 2.0, 0.5)
        _network = competing_destinations(self.pops, self.distances, b=b, c=c, delta=delta)
        # TBD
        raise AssertionError("Test not yet implemented")

    @unittest.skip("Not yet implemented")
    def test_stouffer_exclude_home(self):
        """Test the Stouffer migration function, excluding home."""
        (k, a, b, include_home) = (0.1, 0.5, 1.0, False)
        _network = stouffer(self.pops, self.distances, k=k, a=a, b=b, include_home=include_home)
        # TBD
        raise AssertionError("Test not yet implemented")

    @unittest.skip("Not yet implemented")
    def test_stouffer_include_home(self):
        """Test the Stouffer migration function, excluding home."""
        (k, a, b, include_home) = (0.1, 0.5, 1.0, True)
        _network = stouffer(self.pops, self.distances, k=k, a=a, b=b, include_home=include_home)
        # TBD
        raise AssertionError("Test not yet implemented")

    @unittest.skip("Not yet implemented")
    def test_radiation_exclude_home(self):
        """Test the radiation migration function, excluding home."""
        (k, include_home) = (0.1, False)
        _network = radiation(self.pops, self.distances, k=k, include_home=include_home)
        # TBD
        raise AssertionError("Test not yet implemented")

    @unittest.skip("Not yet implemented")
    def test_radiation_include_home(self):
        """Test the radiation migration function, including home."""
        (k, include_home) = (0.1, True)
        _network = radiation(self.pops, self.distances, k=k, include_home=include_home)
        # TBD
        raise AssertionError("Test not yet implemented")

    def test_distance_one_degree_longitude(self):
        """Test the distance function for one degree of longitude."""
        assert np.isclose(distance(lat1=0, lon1=0, lat2=0, lon2=1), 111.19493, atol=0.00001), print(
            f"1 degree longitude distance is {distance(lat1=0, lon1=0, lat2=0, lon2=1)}, expected 111.19493km"
        )  # 1cm

    def test_distance_one_degree_latitude(self):
        """Test the distance function for one degree of latitude."""
        assert np.isclose(distance(lat1=0, lon1=0, lat2=1, lon2=0), 111.19493, atol=0.00001), print(
            f"1 degree latitude distance is {distance(lat1=0, lon1=0, lat2=1, lon2=0)}, expected 111.19493km"
        )  # 1cm

    def test_distance_nyc_la(self):
        """Test the distance function for New York City to Los Angeles."""
        assert np.isclose(distance(lat1=40.66, lon1=73.94, lat2=34.02, lon2=118.41), 3957.13675, atol=0.00001), print(
            f"NYC to LA distance is {distance(lat1=40.66, lon1=73.94, lat2=34.02, lon2=118.41)}, expected 3957.13675km"
        )  # 1cm

    def test_distance_across_equator(self):
        """Test the distance function for crossing the equator."""
        assert np.isclose(distance(lat1=0.3152, lon1=32.5816, lat2=-1.3032, lon2=36.8474), 507.29393, atol=0.00001), print(
            f"Kampala to Nairobi (crossing the equator) distance is {distance(lat1=0.3152, lon1=32.5816, lat2=-1.3032, lon2=36.8474)}, expected 507.29393km"
        )  # 1cm

        return


if __name__ == "__main__":
    unittest.main()
