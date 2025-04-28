import unittest

import numpy as np

from laser_core.perf import apply
from laser_core.perf import numbafy

class TestPerfUtilities(unittest.TestCase):
    def test_apply(self):
        count = 1_000_000
        velocity = np.random.rand(count) * 20
        np_state = np.zeros(count)
        nb_state = np.zeros(count)
        threshold = 10.0

        def select(i, velocity, threshold):
            return velocity[i] > threshold

        def process(i, state):
            state[i] = 1
            return

        np_state[velocity > threshold] = 1

        apply(select, process, count, velocity=velocity, threshold=threshold, state=nb_state)

        assert np.all(nb_state == np_state), "apply() didn't return the same result as regular NumPy"

        return

    def test_numbafy(self):
        count = 1_000_000
        velocity = np.random.rand(count) * 20
        np_state = np.zeros(count)
        nb_state = np.zeros(count)
        threshold = 10.0

        def select(i, velocity, threshold):
            return velocity[i] > threshold

        def process(i, state):
            state[i] = 1
            # return # intentionally omit `return` statement

        np_state[velocity > threshold] = 1

        update = numbafy(select, process)
        update(count, velocity=velocity, threshold=threshold, state=nb_state)

        assert np.all(nb_state == np_state), "apply() didn't return the same result as regular NumPy"

        return


if __name__ == "__main__":
    unittest.main()
