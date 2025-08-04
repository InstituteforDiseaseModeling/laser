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
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from laser_core import LaserFrame
from laser_core import PropertySet
from laser_core.utils import calc_capacity


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

    def test_add_array_property(self):
        pop = LaserFrame(1024)
        pop.add_array_property("events", (365, 1024))
        assert np.all(pop.events == 0)
        assert pop.events.shape == (365, 1024)

    def test_add_array_property_with_value(self):
        pop = LaserFrame(1024)
        pop.add_array_property("events", (365, 1024), default=42)
        assert np.all(pop.events == 42)
        assert pop.events.shape == (365, 1024)

    def test_add_array_property_with_dtype(self):
        pop = LaserFrame(1024)
        default = np.float32(-3.14159265)
        pop.add_array_property("events", (365, 1024), dtype=np.float32, default=default)
        assert np.all(pop.events == default)
        assert pop.events.shape == (365, 1024)
        assert pop.events.dtype == np.float32

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

        with pytest.raises(
            TypeError, match=re.escape(f"Indices must have the same length as the frame active element count ({pop.count})")
        ):
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

        with pytest.raises(
            TypeError, match=re.escape(f"Indices must have the same length as the frame active element count ({pop.count})")
        ):
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

    def test_save_and_load_snapshot(self):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name

        try:
            # Create frame
            count = 10000
            frame = LaserFrame(capacity=100000, initial_count=count)
            frame.add_scalar_property("age", dtype=np.int32)
            frame.add_scalar_property("status", dtype=np.int8)

            # Assign values
            np.random.seed(42)
            frame.age[:count] = np.random.randint(0, 100, size=count)
            frame.status[:count] = np.random.choice([0, 1], size=count)  # 1 = recovered

            # Squash agents who are recovered or age > 70
            mask = (frame.status == 1) | (frame.age > 70)
            mask = mask[:count]
            removed = mask.sum()
            frame.squash(~mask)

            # Create a 1x10 time series of declining recovered counts
            results_r = np.linspace(removed, 0, 10, dtype=np.float32).reshape(1, -1)

            # Parameters
            pars = PropertySet({"r0": 2.5, "intervention": "vaccine"})

            # Save
            frame.save_snapshot(path, results_r=results_r, pars=pars)

            # Load
            # loaded, r_loaded, pars_loaded = frame.load_snapshot(path, n_ppl=pars["n_ppl"], cbr=pars["cbr"], nt=pars["dur"])
            loaded, r_loaded, pars_loaded = frame.load_snapshot(path, n_ppl=None, cbr=None, nt=None)

            assert loaded.count == frame.count
            assert np.array_equal(loaded.age[: loaded.count], frame.age[: frame.count])
            assert np.array_equal(loaded.status[: loaded.count], frame.status[: frame.count])
            assert np.array_equal(r_loaded, results_r)
            assert pars_loaded["r0"] == 2.5
            # print(f"pars_loaded={pars_loaded}")
            assert pars_loaded["intervention"] == "vaccine"

            # print("test_save_and_load_snapshot passed.")
        finally:
            Path(path).unlink()

    def test_capacity_reasonable_for_various_populations(self):
        for pop_size in [10_000, 100_000, 1_000_000, 10_000_000]:
            with self.subTest(population=pop_size):
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                    path = tmp.name

                try:
                    n_ppl = np.array([pop_size])
                    cbr = 35.0
                    nt = 365 * 3  # 3 years

                    # Simulated frame
                    frame = LaserFrame(capacity=int(pop_size * 1.2), initial_count=pop_size)
                    frame.add_scalar_property("age", dtype=np.int32)
                    frame.add_scalar_property("status", dtype=np.int8)

                    np.random.seed(42)
                    frame.age[:pop_size] = np.random.randint(0, 90, size=pop_size)
                    frame.status[:pop_size] = np.random.choice([0, 1], size=pop_size)

                    # Remove some agents for realism
                    mask = (frame.status == 1) | (frame.age > 75)
                    mask = mask[:pop_size]
                    frame.squash(~mask)

                    results_r = np.linspace(0, 100, 10, dtype=np.float32).reshape(1, -1)
                    frame.save_snapshot(path, results_r=results_r, pars={})

                    # Expected births
                    total_pop = np.sum(n_ppl)
                    expected_final = calc_capacity(total_pop, nt, cbr)
                    expected_births = expected_final - total_pop
                    expected_capacity = frame.count + expected_births

                    # Load
                    loaded, _, _ = LaserFrame.load_snapshot(path, n_ppl=n_ppl, cbr=cbr, nt=nt)

                    # Absolute memory bound
                    bytes_per_agent = 8
                    max_overhead = 10 * 1024 * 1024  # 10 MB
                    excess_agents = loaded.capacity - expected_capacity
                    excess_bytes = excess_agents * bytes_per_agent

                    assert excess_bytes <= max_overhead, (
                        f"Excess memory for pop={pop_size}: "
                        f"{excess_bytes / (1024 * 1024):.2f} MB "
                        f"(capacity={loaded.capacity}, expected={expected_capacity})"
                    )

                finally:
                    Path(path).unlink()

    def test_numpy_ints_for_capacity(self):
        for t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            capacity = t(min(np.iinfo(t).max // 2, 1 << 10))  # Ensure capacity is reasonable
            lf = LaserFrame(capacity)  # just use default initial count
            assert lf.capacity == int(capacity), f"Expected capacity {int(capacity)}, got {lf.capacity}"
            assert lf.count == int(capacity), f"Expected count {int(capacity)}, got {lf.count}"

        return

    def test_numpy_ints_for_initial_count(self):
        for t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            capacity = int(min(np.iinfo(t).max, 1 << 10))  # Ensure capacity is reasonable
            count = t(capacity // 2)
            lf = LaserFrame(capacity, initial_count=count)
            assert lf.capacity == int(capacity), f"Expected capacity {int(capacity)}, got {lf.capacity}"
            assert lf.count == int(count), f"Expected count {int(count)}, got {lf.count}"

        return


if __name__ == "__main__":
    unittest.main()
