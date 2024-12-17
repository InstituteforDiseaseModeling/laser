"""Functions for seeding and accessing the laser-core random number generator.

Using the seed() function here and the pseudo-random number generator (PRNG)
returned from the prng() function in simulation code will guarantee that the
same random number stream is generated and used during simulation runs using
the same seed value (assuming no changes to code which add or remove PRNG calls
or change the number of random draws requested). This is important for
reproducibility and debugging purposes.
"""

from datetime import datetime

import numba as nb
import numpy as np

__all__ = ["get_seed", "prng", "seed"]

_seed: np.uint32 = None
_prng: np.random.Generator = None


@nb.jit((nb.uint32,), nopython=True, nogil=True, parallel=True)
def _nbseed(seed):  # pragma: no cover
    """
    Set the seed for the Numba parallel random number generator (PRNG).

    This function initializes the seed for the non-parallel Numba PRNG and
    then sets the seed for each thread in a parallel execution environment.

    Parameters:

        seed (uint32): The seed value to initialize the PRNG.
    """

    np.random.seed(seed)  # set the non-parallel Numba PRNG seed
    nthreads = nb.get_num_threads()
    for i in nb.prange(nthreads):
        np.random.seed(seed + i)  # actually calls Numba's per-thread PRNG

    return


def seed(seed) -> np.random.Generator:
    """
    Initialize the pseudo-random number generator with a given seed.

    This function sets the global pseudo-random number generator (_prng)
    to a new instance of numpy's default random generator initialized
    with the provided seed. It also seeds Numba's per-thread random number
    generators with the same seed.

    Parameters:

        seed (int): The seed value to initialize the random number generators.

    Returns:

        numpy.random.Generator: The initialized pseudo-random number generator.
    """

    global _seed
    global _prng
    _seed = np.uint32(seed)
    np.random.seed(_seed)
    _prng = np.random.default_rng(_seed)
    _nbseed(np.uint32(_seed))

    return _prng


def get_seed() -> np.uint32:
    """
    Return the seed used to initialize the pseudo-random number generator.

    Returns:

        uint32: The seed value used to initialize the random number generators.
    """

    return _seed


def prng() -> np.random.Generator:
    """Return the global (to LASER) pseudo-random number generator."""
    return _prng if _prng is not None else seed(np.uint32(datetime.now(tz=None).microsecond))  # noqa: DTZ005
