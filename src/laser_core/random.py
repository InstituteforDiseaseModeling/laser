"""Functions for seeding and accessing the laser-core random number generator."""

import numba as nb
import numpy as np

_prng = None


@nb.jit((nb.uint64,), nopython=True, nogil=True, parallel=True)
def nbseed(seed):
    np.random.seed(seed)  # set the non-parallel Numba PRNG seed
    nthreads = nb.get_num_threads()
    for i in nb.prange(nthreads):
        np.random.seed(seed + i)  # actually calls Numba's per-thread PRNG

    return


def seed(seed):
    """
    Initialize the pseudo-random number generator with a given seed.
    This function sets the global pseudo-random number generator (_prng)
    to a new instance of numpy's default random generator initialized
    with the provided seed. It also seeds Numba's per-threadrandom number
    generators with the same seed.
    Parameters:
    seed (int): The seed value to initialize the random number generators.
    Returns:
    numpy.random.Generator: The initialized pseudo-random number generator.
    """

    global _prng
    np.random.seed(seed)
    _prng = np.random.default_rng(seed)
    nbseed(np.uint64(seed))

    return _prng


def prng():
    """Return the global (to LASER) pseudo-random number generator."""
    return _prng
