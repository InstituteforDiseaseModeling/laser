"""Taichi array-based population class for agent-based patch models."""

import taichi as ti


class Population:
    """Taichi array-based Agent Based Population Class"""

    def __init__(self, capacity, **kwargs):
        """Initialize a Population object."""

        self._count = 0
        self._capacity = capacity
        for key, value in kwargs.items():
            setattr(self, key, value)

        return

    def add_property(self, name, dtype=ti.i32, default=0):
        """Add a property to the class"""
        # initialize the property to a Taichi field with of size self._count, dtype, and default value
        setattr(self, name, ti.ndarray(dtype, self._capacity))

        return

    @property
    def count(self):
        return self._count

    @property
    def capacity(self):
        return self._capacity

    def add(self, count: int):
        """Add agents to the population"""
        assert self._count + count <= self._capacity, f"Population exceeds capacity ({self._count=}, {count=}, {self._capacity=})"
        i = self._count
        self._count += int(count)
        j = self._count

        return i, j

    def __len__(self):
        return self._count
