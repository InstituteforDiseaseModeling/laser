"""Single array-based population class for agent-based models"""

from typing import Tuple

import numpy as np


class Population:
    """Array-based Agent Based Population Class"""

    def __init__(self, capacity: int, **kwargs):
        """
        Initialize a Population object.

        Args:
            capacity (int): The maximum capacity of the population (number of agents).
            **kwargs: Additional keyword arguments to set as attributes of the object.

        Returns:
            None
        """
        self._count = 0
        self._capacity = capacity
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__class__.add_property = self.__class__.add_scalar_property  # alias

        return

    # dynamically add a property to the class
    def add_scalar_property(self, name: str, dtype=np.uint32, default=0) -> None:
        """Add a scalar property to the class"""
        # initialize the property to a NumPy array with of size self._count, dtype, and default value
        setattr(self, name, np.full(self._capacity, default, dtype=dtype))
        return

    def add_vector_property(self, name: str, length: int, dtype=np.uint32, default=0) -> None:
        """Add a vector property to the class"""
        # initialize the property to a NumPy array with of size self._count, dtype, and default value
        setattr(self, name, np.full((self._capacity, length), default, dtype=dtype))
        return

    @property
    def count(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, count: int) -> Tuple[int, int]:
        """Add agents to the population"""
        assert self._count + count <= self._capacity, f"Population exceeds capacity ({self._count=}, {count=}, {self._capacity=})"
        i = self._count
        self._count += int(count)
        j = self._count
        return i, j

    def __len__(self) -> int:
        return self._count

    def sort(self, indices, verbose: bool = False) -> None:
        assert np.issubdtype(indices.dtype.type, np.integer), f"Indices must be an integer array (got {indices.dtype})"
        assert (
            indices.shape[0] == self._count
        ), f"Indices ({indices.shape[0]}) must have the same length as the population count {self._count}"
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if verbose:
                    print(f"Sorting {self._count:,} elements of {key}")
                sort = np.zeros_like(value)
                sort[: self._count] = value[indices]
                self.__dict__[key] = sort

        return

    def squash(self, indices, verbose: bool = False) -> None:
        assert indices.dtype == np.dtype("bool"), f"Indices must be a boolean array (got {indices.dtype})"
        current_count = self._count
        assert (
            indices.shape[0] == current_count
        ), f"Indices ({indices.shape[0]}) must have the same length as the population count {current_count}"
        selected_count = indices.sum()
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if verbose:
                    print(f"Squashing {key} from {current_count:,} to {selected_count:,}")
                value[:selected_count] = value[:current_count][indices]
        self._count = selected_count

        return
