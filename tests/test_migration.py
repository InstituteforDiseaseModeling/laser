import csv
import re
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from laser_core.migration import competing_destinations
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import radiation
from laser_core.migration import row_normalizer
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

    def test_distance_with_arrays(self):
        """Test the distance function with arrays."""
        lat1 = np.array([0, 0, 40.66, 0.3152])  # 0, 0, New York, Kampala
        lon1 = np.array([0, 0, 73.94, 32.5816])  # 0, 0, New York, Kampala
        lat2 = np.array([0, 1, 34.02, -1.3032])  # 0, 1, Los Angeles, Nairobi
        lon2 = np.array([1, 0, 118.41, 36.8474])  # 1, 0, Los Angeles, Nairobi
        distances = distance(lat1, lon1, lat2, lon2)
        expected = np.array([111.19493, 111.19493, 3957.13675, 507.29393])
        assert np.allclose(distances, expected, atol=0.00001), print(
            f"distance({lat1=}, {lon1=}, {lat2=}, {lon2=}) = {distances}, expected {expected=}"
        )

        return

    def test_distance_scalar_array(self):
        top_ten = self.city_data[0:10]
        (lat1, lon1) = top_ten[0].lat, top_ten[0].long  # New York
        lat2 = np.array([city.lat for city in top_ten])
        lon2 = np.array([city.long for city in top_ten])
        distances = distance(lat1, lon1, lat2, lon2)
        expected = np.array(
            [0.00000, 3957.13675, 1154.88510, 2283.37528, 3444.88038, 124.08572, 2547.74127, 3906.51957, 2206.50912, 1342.25107]
        )
        assert np.allclose(distances, expected, atol=0.00001), print(
            f"distance({lat1=}, {lon1=}, {lat2=}, {lon2=}) = {distances}, expected {expected=}"
        )

        return

    # TODO? - test with array lat1/lon1 and scalar lat2/lon2
    def test_distance_array_scalar(self):
        top_ten = self.city_data[0:10]
        lat1 = np.array([city.lat for city in top_ten])
        lon1 = np.array([city.long for city in top_ten])
        (lat2, lon2) = top_ten[0].lat, top_ten[0].long  # New York
        distances = distance(lat1, lon1, lat2, lon2)
        expected = np.array(
            [0.00000, 3957.13675, 1154.88510, 2283.37528, 3444.88038, 124.08572, 2547.74127, 3906.51957, 2206.50912, 1342.25107]
        )
        assert np.allclose(distances, expected, atol=0.00001), print(
            f"distance({lat1=}, {lon1=}, {lat2=}, {lon2=}) = {distances}, expected {expected=}"
        )

        return


class TestMigrationFunctionSanityChecks(unittest.TestCase):
    def test_gravity_model_sanity(self):
        """Test the gravity model migration function with maximum."""

        # Test that pop parameter is a NumPy array
        pops = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"pops must be a NumPy array ({type(pops)=})")):
            gravity(pops, np.ones((3, 3)), 1, 1, 1, 1, None)

        # Test that pop parameter is a 1D array
        pops = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a 1D array ({pops.shape=})")):
            gravity(pops, np.ones((3, 3)), 1, 1, 1, 1, None)

        # Test that pop parameter is a numeric array
        pops = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a numeric array ({pops.dtype=})")):
            gravity(pops, np.ones((3, 3)), 1, 1, 1, 1, None)

        # Test that pop parameter is a non-negative array
        with pytest.raises(ValueError, match="pops must contain only non-negative values"):
            gravity(np.array([-1, 2, 3]), np.ones((3, 3)), 1, 1, 1, 1, None)

        # Test that distance parameter is a NumPy array
        distances = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"distances must be a NumPy array ({type(distances)=})")):
            gravity(np.array([1, 2, 3]), distances, 1, 1, 1, 1, None)

        # Test that distance parameter is a 2D array
        distances = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a 2D array ({distances.shape=})")):
            gravity(np.array([1, 2, 3]), distances, 1, 1, 1, 1, None)

        # Test that distance parameter is a square array
        distances = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a square matrix ({distances.shape=})")):
            gravity(np.array([1, 2, 3]), distances, 1, 1, 1, 1, None)

        # Test that distance parameter is a numeric array
        distances = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a numeric array ({distances.dtype=})")):
            gravity(np.array([1, 2, 3]), distances, 1, 1, 1, 1, None)

        # Test that distance parameter is a non-negative array
        with pytest.raises(ValueError, match="distances must contain only non-negative values"):
            gravity(np.array([1, 2, 3]), np.array([[-1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1, 1, 1, 1, None)

        # Test that distance parameter is a symmetric array
        with pytest.raises(ValueError, match="distances must be a symmetric matrix"):
            gravity(np.array([1, 2, 3]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1, 1, 1, 1, None)

        # Test that k parameter is a numeric value
        k = "1"
        with pytest.raises(TypeError, match=re.escape(f"k must be a numeric value ({type(k)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, 1, None)

        # Test that k parameter is a non-negative value
        k = -1
        with pytest.raises(ValueError, match=re.escape(f"k must be a non-negative value ({k=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, 1, None)

        # Test that a parameter is a numeric value
        a = "1"
        with pytest.raises(TypeError, match=re.escape(f"a must be a numeric value ({type(a)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, 1, None)

        # Test that a is a non-negative value
        a = -1
        with pytest.raises(ValueError, match=re.escape(f"a must be a non-negative value ({a=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, None)

        # Test that b parameter is a numeric value
        b = "1"
        with pytest.raises(TypeError, match=re.escape(f"b must be a numeric value ({type(b)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, 1, None)

        # Test that b is a non-negative value
        b = -1
        with pytest.raises(ValueError, match=re.escape(f"b must be a non-negative value ({b=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, 1, None)

        # Test that c parameter is a numeric value
        c = "1"
        with pytest.raises(TypeError, match=re.escape(f"c must be a numeric value ({type(c)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, c, None)

        # Test that c is a non-negative value
        c = -1
        with pytest.raises(ValueError, match=re.escape(f"c must be a non-negative value ({c=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, c, None)

        # Test that max_frac parameter is a numeric value
        max_frac = "1"
        with pytest.raises(TypeError, match=re.escape(f"max_frac must be a numeric value ({type(max_frac)=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, 1, max_frac)

        # Test that max_frac is a non-negative value
        max_frac = -1
        with pytest.raises(ValueError, match=re.escape(f"max_frac must be in [0, 1] ({max_frac=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, 1, max_frac)

        # Test that max_frac is less than or equal to 1
        max_frac = 1.1
        with pytest.raises(ValueError, match=re.escape(f"max_frac must be in [0, 1] ({max_frac=})")):
            gravity(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, 1, 1, max_frac)

        return

    def test_row_normalizer_sanity(self):
        """Test the row normalization function."""

        # Test that network parameter is a NumPy array
        network = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"network must be a NumPy array ({type(network)=})")):
            row_normalizer(network, 0.1)

        # Test that network parameter is a numeric array
        network = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"network must be a numeric array ({network.dtype=})")):
            row_normalizer(network, 0.1)

        # Test that network parameter is a 2D array
        network = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"network must be a 2D array ({network.shape=})")):
            row_normalizer(network, 0.1)

        # Test that network parameter is a square array
        network = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(TypeError, match=re.escape(f"network must be a square matrix ({network.shape=})")):
            row_normalizer(network, 0.1)

        # Test that network parameter is a non-negative array
        with pytest.raises(ValueError, match="network must contain only non-negative values"):
            row_normalizer(np.array([[-1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0.1)

        # Test that max_rowsum parameter is a numeric value
        max_rowsum = "0.1"
        with pytest.raises(TypeError, match=re.escape(f"max_rowsum must be a numeric value ({type(max_rowsum)=})")):
            row_normalizer(np.full((3, 3), 0.0625), max_rowsum)

        # Test that max_rowsum is a non-negative value
        with pytest.raises(ValueError, match=re.escape("max_rowsum must be in [0, 1]")):
            row_normalizer(np.full((3, 3), 0.0625), -0.1)

        # Test that max_rowsum is less than or equal to 1
        with pytest.raises(ValueError, match=re.escape("max_rowsum must be in [0, 1]")):
            row_normalizer(np.full((3, 3), 0.0625), 1.1)

        return

    def test_competing_destinations_sanity(self):
        """Test the competing destinations migration function."""

        # Test that pop parameter is a NumPy array
        pops = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"pops must be a NumPy array ({type(pops)=})")):
            competing_destinations(pops, np.ones((3, 3)), 1, 1, 1)

        # Test that pop parameter is a 1D array
        pops = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a 1D array ({pops.shape=})")):
            competing_destinations(pops, np.ones((3, 3)), 1, 1, 1)

        # Test that pop parameter is a numeric array
        pops = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a numeric array ({pops.dtype=})")):
            competing_destinations(pops, np.ones((3, 3)), 1, 1, 1)

        # Test that pop parameter is a non-negative array
        with pytest.raises(ValueError, match="pops must contain only non-negative values"):
            competing_destinations(np.array([-1, 2, 3]), np.ones((3, 3)), 1, 1, 1)

        # Test that distance parameter is a NumPy array
        distances = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"distances must be a NumPy array ({type(distances)=})")):
            competing_destinations(np.array([1, 2, 3]), distances, 1, 1, 1)

        # Test that distance parameter is a 2D array
        distances = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a 2D array ({distances.shape=})")):
            competing_destinations(np.array([1, 2, 3]), distances, 1, 1, 1)

        # Test that distance parameter is a square array
        distances = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a square matrix ({distances.shape=})")):
            competing_destinations(np.array([1, 2, 3]), distances, 1, 1, 1)

        # Test that distance parameter is a numeric array
        distances = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a numeric array ({distances.dtype=})")):
            competing_destinations(np.array([1, 2, 3]), distances, 1, 1, 1)

        # Test that distance parameter is a non-negative array
        with pytest.raises(ValueError, match="distances must contain only non-negative values"):
            competing_destinations(np.array([1, 2, 3]), np.array([[-1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1, 1, 1)

        # Test that parameter b is a numeric value
        b = "1"
        with pytest.raises(TypeError, match=re.escape(f"b must be a numeric value ({type(b)=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), b, 1, 1)

        # Test that b is a non-negative value
        b = -1
        with pytest.raises(ValueError, match=re.escape(f"b must be a non-negative value ({b=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), b, 1, 1)

        # Test that parameter c is a numeric value
        c = "1"
        with pytest.raises(TypeError, match=re.escape(f"c must be a numeric value ({type(c)=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, c, 1)

        # Test that c is a non-negative value
        c = -1
        with pytest.raises(ValueError, match=re.escape(f"c must be a non-negative value ({c=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, c, 1)

        # Test that parameter delta is a numeric value
        delta = "1"
        with pytest.raises(TypeError, match=re.escape(f"delta must be a numeric value ({type(delta)=})")):
            competing_destinations(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, delta)

        return

    def test_stouffer_sanity(self):
        """Test the Stouffer migration function."""

        # Test that pop parameter is a NumPy array
        pops = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"pops must be a NumPy array ({type(pops)=})")):
            stouffer(pops, np.ones((3, 3)), 1, 1, 1, False)

        # Test that pop parameter is a 1D array
        pops = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a 1D array ({pops.shape=})")):
            stouffer(pops, np.ones((3, 3)), 1, 1, 1, False)

        # Test that pop parameter is a numeric array
        pops = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a numeric array ({pops.dtype=})")):
            stouffer(pops, np.ones((3, 3)), 1, 1, 1, False)

        # Test that pop parameter is a non-negative array
        with pytest.raises(ValueError, match="pops must contain only non-negative values"):
            stouffer(np.array([-1, 2, 3]), np.ones((3, 3)), 1, 1, 1, False)

        # Test that distance parameter is a NumPy array
        distances = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"distances must be a NumPy array ({type(distances)=})")):
            stouffer(np.array([1, 2, 3]), distances, 1, 1, 1, False)

        # Test that distance parameter is a 2D array
        distances = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a 2D array ({distances.shape=})")):
            stouffer(np.array([1, 2, 3]), distances, 1, 1, 1, False)

        # Test that distance parameter is a square array
        distances = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a square matrix ({distances.shape=})")):
            stouffer(np.array([1, 2, 3]), distances, 1, 1, 1, False)

        # Test that distance parameter is a numeric array
        distances = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a numeric array ({distances.dtype=})")):
            stouffer(np.array([1, 2, 3]), distances, 1, 1, 1, False)

        # Test that parameter k is a numeric value
        k = "1"
        with pytest.raises(TypeError, match=re.escape(f"k must be a numeric value ({type(k)=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, False)

        # Test that k is a non-negative value
        k = -1
        with pytest.raises(ValueError, match=re.escape(f"k must be a non-negative value ({k=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), k, 1, 1, False)

        # Test that parameter a is a numeric value
        a = "1"
        with pytest.raises(TypeError, match=re.escape(f"a must be a numeric value ({type(a)=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, False)

        # Test that a is a non-negative value
        a = -1
        with pytest.raises(ValueError, match=re.escape(f"a must be a non-negative value ({a=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), 1, a, 1, False)

        # Test that parameter b is a numeric value
        b = "1"
        with pytest.raises(TypeError, match=re.escape(f"b must be a numeric value ({type(b)=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, False)

        # Test that b is a non-negative value
        b = -1
        with pytest.raises(ValueError, match=re.escape(f"b must be a non-negative value ({b=})")):
            stouffer(np.array([1, 2, 3]), np.ones((3, 3)), 1, 1, b, False)

        return

    def test_radiation_sanity(self):
        """Test the radiation migration function."""

        # Test that pop parameter is a NumPy array
        pops = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"pops must be a NumPy array ({type(pops)=})")):
            radiation(pops, np.ones((3, 3)), 1, False)

        # Test that pop parameter is a 1D array
        pops = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a 1D array ({pops.shape=})")):
            radiation(pops, np.ones((3, 3)), 1, False)

        # Test that pop parameter is a numeric array
        pops = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match=re.escape(f"pops must be a numeric array ({pops.dtype=})")):
            radiation(pops, np.ones((3, 3)), 1, False)

        # Test that pop parameter is a non-negative array
        with pytest.raises(ValueError, match="pops must contain only non-negative values"):
            radiation(np.array([-1, 2, 3]), np.ones((3, 3)), 1, False)

        # Test that distance parameter is a NumPy array
        distances = [1, 2, 3]
        with pytest.raises(TypeError, match=re.escape(f"distances must be a NumPy array ({type(distances)=})")):
            radiation(np.array([1, 2, 3]), distances, 1, False)

        # Test that distance parameter is a 2D array
        distances = np.array([1, 2, 3])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a 2D array ({distances.shape=})")):
            radiation(np.array([1, 2, 3]), distances, 1, False)

        # Test that distance parameter is a square array
        distances = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a square matrix ({distances.shape=})")):
            radiation(np.array([1, 2, 3]), distances, 1, False)

        # Test that distance parameter is a numeric array
        distances = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
        with pytest.raises(TypeError, match=re.escape(f"distances must be a numeric array ({distances.dtype=})")):
            radiation(np.array([1, 2, 3]), distances, 1, False)

        # Test that parameter k is a numeric value
        k = "1"
        with pytest.raises(TypeError, match=re.escape(f"k must be a numeric value ({type(k)=})")):
            radiation(np.array([1, 2, 3]), np.ones((3, 3)), k, False)

        # Test that k is a non-negative value
        k = -1
        with pytest.raises(ValueError, match=re.escape(f"k must be a non-negative value ({k=})")):
            radiation(np.array([1, 2, 3]), np.ones((3, 3)), k, False)

        return

    def test_distance_sanity(self):
        """Test the distance function."""

        # Test that lat1 parameter is a numeric value
        with pytest.raises(TypeError, match="lat1 must be a numeric value or NumPy array"):
            distance("0", 0, 0, 0)

        # Test that lat1 is a valid latitude
        with pytest.raises(ValueError, match=re.escape("lat1 must be in the range [-90, 90]")):
            distance(91, 0, 0, 0)

        # Test that lon1 parameter is a numeric value
        with pytest.raises(TypeError, match="lon1 must be a numeric value or NumPy array"):
            distance(0, "0", 0, 0)

        # Test that lon1 is a valid longitude
        with pytest.raises(ValueError, match=re.escape("lon1 must be in the range [-180, 180]")):
            distance(0, 181, 0, 0)

        # Test that lat2 parameter is a numeric value
        with pytest.raises(TypeError, match="lat2 must be a numeric value or NumPy array"):
            distance(0, 0, "0", 0)

        # Test that lat2 is a valid latitude
        with pytest.raises(ValueError, match=re.escape("lat2 must be in the range [-90, 90]")):
            distance(0, 0, 91, 0)

        # Test that lon2 parameter is a numeric value
        with pytest.raises(TypeError, match="lon2 must be a numeric value or NumPy array"):
            distance(0, 0, 0, "0")

        # Test that lon2 is a valid longitude
        with pytest.raises(ValueError, match=re.escape("lon2 must be in the range [-180, 180]")):
            distance(0, 0, 0, 181)

        return


if __name__ == "__main__":
    unittest.main()
