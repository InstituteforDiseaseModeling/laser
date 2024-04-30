"""Single array-based population class for agent-based models"""

import numpy as np


class Population:
    """Array-based Agent Based Population Class"""

    def __init__(self, capacity, **kwargs):
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
        return

    # dynamically add a property to the class
    def add_property(self, name, dtype=np.uint32, default=0):
        """Add a property to the class"""
        # initialize the property to a NumPy array with of size self._count, dtype, and default value
        setattr(self, name, np.full(self._capacity, default, dtype=dtype))
        return

    @property
    def count(self):
        return self._count

    @property
    def capacity(self):
        return self._capacity

    def add(self, count):
        """Add agents to the population"""
        assert self._count + count <= self._capacity, f"Population exceeds capacity ({self._count=}, {count=}, {self._capacity=})"
        i = self._count
        self._count += count
        j = self._count
        return i, j

    def __len__(self):
        return self._count
