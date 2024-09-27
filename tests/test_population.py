import unittest

import numpy as np

from idmlaser.numpynumba import Population


class TestPopulation(unittest.TestCase):
    def test_init(self):
        pop = Population(1024)
        assert pop.capacity == 1024
        assert pop.count == 0
        assert len(pop) == pop.count

    def test_init_with_properties(self):
        pop = Population(1024, start_year=1944, source="https://ourworldindata.org/grapher/life-expectancy?country=~USA")
        assert pop.capacity == 1024
        assert pop.count == 0
        assert len(pop) == pop.count
        assert pop.start_year == 1944
        assert pop.source == "https://ourworldindata.org/grapher/life-expectancy?country=~USA"

    def test_add_scalar_property(self):
        pop = Population(1024)
        pop.add_scalar_property("age", default=0)
        assert pop.age[0] == 0
        assert pop.age.shape == (1024,)

    def test_add_scalar_property_with_value(self):
        pop = Population(1024)
        pop.add_scalar_property("age", default=10)
        assert pop.age[0] == 10
        assert pop.age.shape == (1024,)

    # deprecated `add_property` method, should use `add_scalar_property`
    def test_add_property(self):
        pop = Population(1024)
        pop.add_scalar_property("age", default=0)
        assert pop.age[0] == 0
        assert pop.age.shape == (1024,)

    # deprecated `add_property` method, should use `add_scalar_property`
    def test_add_property_with_value(self):
        pop = Population(1024)
        pop.add_scalar_property("age", default=10)
        assert pop.age[0] == 10
        assert pop.age.shape == (1024,)

    def test_add_vector_property(self):
        pop = Population(1024)
        pop.add_vector_property("events", 365)
        assert pop.events[0, 0] == 0
        assert pop.events.shape == (1024, 365)

    def test_add_agents(self):
        pop = Population(1024)
        istart, iend = pop.add(100)
        assert istart == 0
        assert iend == 100
        assert pop.count == 100
        assert len(pop) == pop.count

    def test_add_agents_again(self):
        pop = Population(1024)
        istart, iend = pop.add(100)
        assert istart == 0
        assert iend == 100
        assert pop.count == 100
        assert len(pop) == pop.count

        istart, iend = pop.add(100)
        assert istart == 100
        assert iend == 200
        assert pop.count == 200
        assert len(pop) == pop.count

    def test_sort(self):
        pop = Population(1024)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart, iend = pop.add(100)
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)  # random ages 0-100 years
        original_age = np.array(pop.age[: pop.count])
        pop.height[istart:iend] = np.random.default_rng().uniform(0.5, 2.0, 100)  # random heights 0.5-2 meters
        original_height = np.array(pop.height[: pop.count])
        indices = np.argsort(pop.age[: pop.count])
        pop.sort(indices, verbose=True)
        assert np.all(pop.age[: pop.count] == np.sort(original_age))
        assert np.all(pop.height[: pop.count] == original_height[indices])

    def test_squash(self):
        pop = Population(1024)
        pop.add_scalar_property("age", default=0)
        pop.add_scalar_property("height", default=0.0, dtype=np.float32)
        istart, iend = pop.add(100)
        pop.age[istart:iend] = np.random.default_rng().integers(0, 100, 100)  # random ages 0-100 years
        original_age = np.array(pop.age[: pop.count])
        pop.height[istart:iend] = np.random.default_rng().uniform(0.5, 2.0, 100)  # random heights 0.5-2 meters
        original_height = np.array(pop.height[: pop.count])
        keep = pop.age[: pop.count] >= 40
        pop.squash(keep, verbose=True)
        assert pop.count == keep.sum()
        assert np.all(pop.age[: pop.count] == original_age[keep])
        assert np.all(pop.height[: pop.count] == original_height[keep])


if __name__ == "__main__":
    unittest.main()
