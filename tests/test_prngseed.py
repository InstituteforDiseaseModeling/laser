import unittest

import numba as nb
import numpy as np

from laser_core.random import prng as rng
from laser_core.random import seed


# Make some calls to various random number generators with Numba threads.
@nb.njit((nb.uint64, nb.int64[:], nb.float64[:], nb.float64[:], nb.uint64[:]), nogil=True, parallel=True)
def noise(count, integers, floats, normal, poisson):
    for i in nb.prange(count):
        integers[i] = np.random.randint(0, 100)
        floats[i] = np.random.random()
        normal[i] = np.random.normal()
        poisson[i] = np.random.poisson(5)

    return


# Make some calls to the Numba-fied random number generator in non-parallel mode.
@nb.njit((nb.uint64, nb.int64[:]), nogil=True)
def nbintegers(count, integers):
    for i in range(count):
        integers[i] = np.random.randint(0, 100)

    return


class TestRandomSeed(unittest.TestCase):
    def test_numpy_seed(self):
        """Test that setting the random seed applies to the built-in NumPy generator."""
        np.random.seed(20241009)
        integers1 = np.random.randint(0, 100, 10)
        floats1 = np.random.random(10)
        normal1 = np.random.normal(0, 1, 10)
        poisson1 = np.random.poisson(5, 10)
        np.random.seed(20241009)
        integers2 = np.random.randint(0, 100, 10)
        floats2 = np.random.random(10)
        normal2 = np.random.normal(0, 1, 10)
        poisson2 = np.random.poisson(5, 10)

        assert np.array_equal(integers1, integers2)
        assert np.allclose(floats1, floats2)
        assert np.allclose(normal1, normal2)
        assert np.array_equal(poisson1, poisson2)

        return

    def test_prng_seed(self):
        """Test that setting the random seed applies to the laser-core "global" prng."""
        prng = seed(20241009)
        integers1 = prng.integers(0, 100, 10)
        floats1 = prng.random(10)
        normal1 = prng.normal(0, 1, 10)
        poisson1 = prng.poisson(5, 10)
        prng = seed(20241009)
        integers2 = prng.integers(0, 100, 10)
        floats2 = prng.random(10)
        normal2 = prng.normal(0, 1, 10)
        poisson2 = prng.poisson(5, 10)

        assert np.array_equal(integers1, integers2)
        assert np.allclose(floats1, floats2)
        assert np.allclose(normal1, normal2)
        assert np.array_equal(poisson1, poisson2)

        return

    def test_rng_seed(self):
        """Test that setting the random seed applies equally to the prng returned from `seed()` and from `rng()`."""
        prng = seed(20241009)
        integers1 = prng.integers(0, 100, 10)
        floats1 = prng.random(10)
        normal1 = prng.normal(0, 1, 10)
        poisson1 = prng.poisson(5, 10)
        _ = seed(20241009)
        prng = rng()
        integers2 = prng.integers(0, 100, 10)
        floats2 = prng.random(10)
        normal2 = prng.normal(0, 1, 10)
        poisson2 = prng.poisson(5, 10)

        assert np.array_equal(integers1, integers2)
        assert np.allclose(floats1, floats2)
        assert np.allclose(normal1, normal2)
        assert np.array_equal(poisson1, poisson2)

        return

    def test_numba_seed(self):
        """Test that setting the random seed applies to Numba-fied functions."""
        count = 1024
        seed(20241009)
        integers1 = np.empty(count, dtype=np.int64)
        floats1 = np.empty(count, dtype=np.float64)
        normal1 = np.empty(count, dtype=np.float64)
        poisson1 = np.empty(count, dtype=np.uint64)
        noise(np.uint64(count), integers1, floats1, normal1, poisson1)
        integers1a = np.array(integers1)
        nbintegers(1024, integers1a)
        seed(20241009)
        integers2 = np.empty(count, dtype=np.int64)
        floats2 = np.empty(count, dtype=np.float64)
        normal2 = np.empty(count, dtype=np.float64)
        poisson2 = np.empty(count, dtype=np.uint64)
        noise(np.uint64(count), integers2, floats2, normal2, poisson2)
        integers2a = np.array(integers2)
        nbintegers(1024, integers2a)

        assert np.array_equal(integers1, integers2)
        assert np.allclose(floats1, floats2)
        assert np.allclose(normal1, normal2)
        assert np.array_equal(poisson1, poisson2)
        assert np.array_equal(integers1a, integers2a)

        return


if __name__ == "__main__":
    unittest.main()
