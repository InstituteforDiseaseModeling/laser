"""Single array-based population class for agent-based models"""

from collections.abc import Iterable
from typing import Tuple
from typing import Union

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

        self.add_property = self.add_scalar_property  # alias

        return

    # dynamically add a property to the class
    def add_scalar_property(self, name: str, dtype=np.uint32, default=0) -> None:
        """Add a scalar property to the class"""
        # initialize the property to a NumPy array with of size self._count, dtype, and default value
        setattr(self, name, np.full(self._capacity, default, dtype=dtype))
        return

    def add_vector_property(self, name, dims: Union[int, Tuple, list], dtype=np.uint32, default=0) -> None:
        """Add a vector property to the class"""
        # initialize the property to a NumPy array with of size self._count, dtype, and default value
        dimensions = [self._capacity]
        if isinstance(dims, Iterable):
            dimensions.extend(dims)
        else:
            dimensions.append(dims)
        setattr(self, name, np.full(dimensions, default, dtype=dtype))
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
