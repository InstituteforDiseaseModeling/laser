"""
Unit tests for the LaserFrame class in the laser_core.laserframe module.

This module contains a series of unit tests for the LaserFrame class, which is
designed to manage a collection of agents with various properties. The tests
cover initialization, property addition, agent addition, sorting, and squashing
functionality.

Classes:
    TestLaserFrame: A unittest.TestCase subclass that contains tests for the
    LaserFrame class.

Test Methods:
    - test_init: Tests the initialization of a LaserFrame instance with a
      specified capacity.
    - test_init_with_properties: Tests the initialization of a LaserFrame
      instance with additional properties.
    - test_add_scalar_property: Tests the addition of a scalar property with a
      default value.
    - test_add_scalar_property_with_value: Tests the addition of a scalar
      property with a specified default value.
    - test_add_property: Tests the deprecated add_property method (should use
      add_scalar_property).
    - test_add_property_with_value: Tests the deprecated add_property method
      with a specified default value (should use add_scalar_property).
    - test_add_vector_property: Tests the addition of a vector property with a
      specified length.
    - test_add_agents: Tests the addition of agents to the LaserFrame.
    - test_add_agents_again: Tests the addition of agents to the LaserFrame
      multiple times.
    - test_sort: Tests the sorting of agents based on a scalar property.
    - test_squash: Tests the squashing (filtering) of agents based on a
      condition.

Usage:
    Run this module with a Python interpreter to execute the unit tests.
"""

import re
import unittest

import numpy as np
import pytest

from laser_core import LaserFrame


class TestLaserFrame(unittest.TestCase):
    def test_init(self):
        pop = LaserFrame(1024, initial_count=0)
        assert pop.capacity == 1024
        assert pop.count == 0
        assert len(pop) == pop.count

    def test_init_with_count(self):
        pop = LaserFrame(1024, initial_count=128)
        assert pop.capacity == 1024
        assert pop.count == 128
        assert len(pop) == pop.count

    def test_init_with_count_minus_one(self):
        pop = LaserFrame(1024, initial_count=-1)
        assert pop.capacity == 1024
        assert pop.count == 1024
        assert len(pop) == pop.count

    def test_init_with_properties(self):
        pop = LaserFrame(1024, initial_count=0, start_year=1944, source="https://ourworldindata.org/grapher/life-expectancy?country=~USA")
        assert pop.capacity == 1024
        assert pop.count == 0
        assert len(pop) == pop.count
        assert pop.start_year == 1944
        assert pop.source == "https://ourworldindata.org/grapher/life-expectancy?country=~USA"

    def test_add_scalar_property(self):
        pop = LaserFrame(1024)
        pop.add_scalar_property("age", default=0)
        assert np.all(pop.age == 0)
        assert pop.age.shape == (1024,)

    def test_add_scalar_property_with_value(self):
        pop = LaserFrame(1024)
        pop.add_scalar_property("age", default=10)
        assert np.all(pop.age == 10)
        assert pop.age.shape == (1024,)

    def test_add_vector_property(self):
        pop = LaserFrame(1024)
        pop.add_vector_property("events", 365)
        assert np.all(pop.events == 0)
        assert pop.events.shape == (365, 1024)

    def test_add_vectory_property_with_value(self):
        pop = LaserFrame(1024)
        pop.add_vector_property("events", 365, default=1)
        assert np.all(pop.events == 1)
        assert pop.events.shape == (365, 1024)

    def test_add_agents(self):
        pop = LaserFrame(1024, 100)
        assert pop.count == 100
        assert len(pop) == pop.count
        istart, iend = pop.add(200)
        assert istart == 100
        assert iend == 300
        assert pop.count == 300
        assert len(pop) == pop.count

    def test_add_agents_again(self):
        pop = LaserFrame(1024, 100)
        istart, iend = pop.add(200)
        istart, iend = pop.add(500)
        assert istart == 300
        assert iend == 800
        assert pop.count == 800
        assert len(pop) == pop.count

    def test_add_too_many_agents(self):
        pop = LaserFrame(1024, 1000)
        assert pop.count == 1000
        assert len(pop) == pop.count

        with pytest.raises(
            ValueError, match=re.escape("frame.add() exceeds capacity (self._count=1000 + count=100 > self._capacity=1024)")
        ):
            pop.add(100)

    def test_sort(self):
        pop = LaserFrame(1024, initial_count=100)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart = 0
        iend = pop.count
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)  # random ages 0-100 years
        original_age = np.array(pop.age[: pop.count])
        pop.height[istart:iend] = np.random.default_rng().uniform(0.5, 2.0, 100)  # random heights 0.5-2 meters
        original_height = np.array(pop.height[: pop.count])
        indices = np.argsort(pop.age[: pop.count])
        pop.sort(indices, verbose=False)
        assert np.all(pop.age[: pop.count] == np.sort(original_age))
        assert np.all(pop.height[: pop.count] == original_height[indices])

    def test_sort_sanity_check(self):
        pop = LaserFrame(1024, initial_count=100)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart = 0
        iend = pop.count
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)
        indices = np.argsort(pop.age[: pop.count])

        with pytest.raises(TypeError, match=re.escape(f"Indices must be a numpy array (got {list})")):
            pop.sort(indices.tolist(), verbose=True)

        with pytest.raises(TypeError, match=re.escape(f"Indices must have the same length as the population count ({pop.count})")):
            pop.sort(indices[0:50], verbose=True)

        with pytest.raises(TypeError, match=re.escape("Indices must be an integer array (got float32)")):
            pop.sort(indices.astype(np.float32), verbose=1)

    def test_squash(self):
        pop = LaserFrame(1024, initial_count=100)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart = 0
        iend = pop.count
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)  # random ages 0-100 years
        original_age = np.array(pop.age[: pop.count])
        pop.height[istart:iend] = np.random.default_rng().uniform(0.5, 2.0, 100)  # random heights 0.5-2 meters
        original_height = np.array(pop.height[: pop.count])
        keep = pop.age[: pop.count] >= 40
        pop.squash(keep, verbose=False)
        assert pop.count == keep.sum()
        assert np.all(pop.age[: pop.count] == original_age[keep])
        assert np.all(pop.height[: pop.count] == original_height[keep])

    def test_squash_sanity_checks(self):
        pop = LaserFrame(1024, initial_count=100)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart = 0
        iend = 100
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)
        keep = pop.age[: pop.count] >= 40

        with pytest.raises(TypeError, match=re.escape(f"Indices must be a numpy array (got {list})")):
            pop.squash(keep.tolist(), verbose=True)

        with pytest.raises(TypeError, match=re.escape(f"Indices must have the same length as the population count ({pop.count})")):
            pop.squash(keep[0:50], verbose=True)

        with pytest.raises(TypeError, match=re.escape("Indices must be a boolean array (got float32)")):
            pop.squash(keep.astype(np.float32), verbose=1)

    def test_init_bad_capacity1(self):
        capacity = "5150"
        with pytest.raises(ValueError, match=re.escape(f"Capacity must be a positive integer, got {capacity}.")):
            _ = LaserFrame(capacity=capacity)

    def test_init_bad_capacity2(self):
        capacity = -5150
        with pytest.raises(ValueError, match=re.escape(f"Capacity must be a positive integer, got {capacity}.")):
            _ = LaserFrame(capacity=capacity)

    def test_init_bad_initial_count1(self):
        initial_count = "5150"
        with pytest.raises(ValueError, match=re.escape(f"Initial count must be a non-negative integer, got {initial_count}.")):
            _ = LaserFrame(capacity=65536, initial_count=initial_count)

    def test_init_bad_initial_count2(self):
        initial_count = -5150
        with pytest.raises(ValueError, match=re.escape(f"Initial count must be a non-negative integer, got {initial_count}.")):
            _ = LaserFrame(capacity=65536, initial_count=initial_count)

    def test_init_bad_initial_count3(self):
        capacity = 65536
        initial_count = 1_000_000
        with pytest.raises(ValueError, match=re.escape(f"Initial count ({initial_count}) cannot exceed capacity ({capacity}).")):
            _ = LaserFrame(capacity=capacity, initial_count=initial_count)


if __name__ == "__main__":
    unittest.main()
