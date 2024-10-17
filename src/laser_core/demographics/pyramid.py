"""A class for generating samples from a distribution using the Vose alias method."""

from pathlib import Path

import numpy as np

from laser_core.random import prng


class AliasedDistribution:
    """A class to generate samples from a distribution using the Vose alias method."""

    def __init__(self, counts):
        # TODO, consider int64 or uint64 if using global population
        alias = np.full(len(counts), -1, dtype=np.int32)
        probs = np.array(counts, dtype=np.int32)
        total = probs.sum()
        probs *= len(probs)  # See following explanation.

        """
        We want to know if any particular bin's probability is less than the average probability.
        So we want to know if p[i] < (p.sum() / len(p)).
        We rearrange this to (p[i] * len(p)) < p.sum().
        Now we can check below if p'[i] < p.sum() where p'[i] = p[i] * len(p).
        """

        small = [i for i, value in enumerate(probs) if value < total]
        large = [i for i, value in enumerate(probs) if value > total]
        while small:
            ismall = small.pop()
            ilarge = large.pop()
            alias[ismall] = ilarge
            probs[ilarge] -= total - probs[ismall]
            if probs[ilarge] < total:
                small.append(ilarge)
            elif probs[ilarge] > total:
                large.append(ilarge)

        self._alias = alias
        self._probs = probs
        self._total = total

        return

    @property
    def alias(self) -> np.ndarray:
        return self._alias

    @property
    def probs(self) -> np.ndarray:
        return self._probs

    @property
    def total(self) -> int:
        return self._total

    def sample(self, count=1) -> int:
        """Generate samples from the distribution."""

        rng = prng()

        if count == 1:
            i = rng.integers(low=0, high=len(self._alias))
            d = rng.integers(low=0, high=self._total)

            i = i if d < self._probs[i] else self._alias[i]
        else:
            i = rng.integers(low=0, high=len(self._alias), size=count)
            d = rng.integers(low=0, high=self._total, size=count)
            a = d >= self._probs[i]
            i[a] = self._alias[i[a]]

        return i


def load_pyramid_csv(file: Path, quiet=False) -> np.ndarray:
    """Load a CSV file with population pyramid data."""

    if not quiet:
        print(f"Reading population pyramid data from '{file}' ...")
    # Expected file schema:
    # "Age,M,F"
    # "low-high,#males,#females"
    # ...
    # "max+,#males,#females"
    with file.open("r") as f:
        # Use strip to remove newline characters
        lines = [line.strip() for line in f.readlines()]
    text = lines[1:]  # Skip the first line
    text = [line.split(",") for line in text]  # Split each line by comma
    # Split the first element by hyphen
    text = [line[0].split("-") + line[1:] for line in text]
    # Remove the plus sign from the last element
    text[-1][0] = text[-1][0].replace("+", "")
    data = [list(map(int, line)) for line in text]  # Convert all elements to integers
    data[-1] = [
        data[-1][0],
        data[-1][0],
        *data[-1][1:],
    ]  # Make the last element a single year bucket

    datanp = np.zeros((len(data), 5), dtype=np.int32)
    for i, line in enumerate(data):
        datanp[i, :4] = line
    datanp[:, 4] = datanp[:, 2] + datanp[:, 3]  # Total population (male + female)

    return datanp
