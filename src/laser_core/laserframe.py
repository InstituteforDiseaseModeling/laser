"""
laserframe.py

This module defines the LaserFrame class, which is used to manage dynamically allocated data for agents or nodes/patches.
The LaserFrame class is similar to a database table or a Pandas DataFrame and supports scalar and vector properties.

Classes:
    LaserFrame: A class to manage dynamically allocated data for agents or nodes/patches.

Usage Example:

.. code-block:: python

    laser_frame = LaserFrame(capacity=100)
    laser_frame.add_scalar_property('age', dtype=np.int32, default=0)
    laser_frame.add_vector_property('position', length=3, dtype=np.float32, default=0.0)
    start, end = laser_frame.add(10)
    laser_frame.sort(np.arange(10)[::-1])
    laser_frame.squash(np.array([True, False, True, False, True, False, True, False, True, False]))

Attributes:
    _count (int): The current count of agents.
    _capacity (int): The maximum capacity of the population.
"""

import numpy as np


class LaserFrame:
    """
    The LaserFrame class, similar to a db table or a Pandas DataFrame, holds dynamically
    allocated data for agents (generally 1-D or scalar) or for nodes|patches (e.g., 1-D for
    scalar value per patch or 2-D for time-varying per patch)."""

    def __init__(self, capacity: int, **kwargs):
        """
        Initialize a LaserFrame object.

        Parameters:

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
        """
        Add a scalar property to the class.

        This method initializes a new scalar property for the class instance. The property is
        stored as a NumPy array with a specified data type and default value.

        Parameters:

            name (str): The name of the scalar property to be added.
            dtype (data-type, optional): The desired data type for the property. Default is np.uint32.
            default (scalar, optional): The default value for the property. Default is 0.

        Returns:

            None
        """

        # initialize the property to a NumPy array with of size self._capacity, dtype, and default value
        setattr(self, name, np.full(self._capacity, default, dtype=dtype))
        return

    def add_vector_property(self, name: str, length: int, dtype=np.uint32, default=0) -> None:
        """
        Adds a vector property to the object.

        This method initializes a new property with the given name as a NumPy array.

        The array will have a shape of (length, self._capacity) and will be filled
        with the specified default value. The data type of the array elements is
        determined by the `dtype` parameter.

        Parameters:

            name (str): The name of the property to be added.
            length (int): The length of the vector.
            dtype (data-type, optional): The desired data-type for the array, default is np.uint32.
            default (scalar, optional): The default value to fill the array with, default is 0.

        Returns:

            None
        """

        # initialize the property to a NumPy array with of size (length, self._capacity), dtype, and default value
        setattr(self, name, np.full((length, self._capacity), default, dtype=dtype))
        return

    @property
    def count(self) -> int:
        """
        Returns the current count (equivalent to len()).

        Returns:

            int: The current count value.
        """

        return self._count

    @property
    def capacity(self) -> int:
        """
        Returns the capacity of the laser frame (total possible entries for dynamic properties).

        Returns:

            int: The capacity of the laser frame.
        """

        return self._capacity

    def add(self, count: int) -> tuple[int, int]:
        """
        Adds the specified count to the current count of the LaserFrame.

        This method increments the internal count by the given count, ensuring that the total does not exceed the frame's capacity. If the addition would exceed the capacity, an assertion error is raised.

        Parameters:

            count (int): The number to add to the current count.

        Returns:

            tuple[int, int]: A tuple containing the [start index, end index) after the addition.

        Raises:

            AssertionError: If the resulting count exceeds the frame's capacity.
        """

        if not self._count + count <= self._capacity:
            raise ValueError(f"frame.add() exceeds capacity ({self._count=} + {count=} > {self._capacity=})")

        i = self._count
        self._count += int(count)
        j = self._count
        return i, j

    def __len__(self) -> int:
        return self._count

    def sort(self, indices, verbose: bool = False) -> None:
        """
        Sorts the elements of the object's numpy arrays based on the provided indices.

        Parameters:

            indices (np.ndarray): An array of indices used to sort the numpy arrays. Must be of integer type and have the same length as the population count (`self._count`).

            verbose (bool, optional): If True, prints the sorting progress for each numpy array attribute. Defaults to False.

        Raises:

            AssertionError: If `indices` is not an integer array or if its length does not match the population count.
        """

        _is_instance(indices, np.ndarray, f"Indices must be a numpy array (got {type(indices)})")
        _has_shape(indices, (self._count,), f"Indices must have the same length as the population count ({self._count})")
        _is_dtype(indices, np.integer, f"Indices must be an integer array (got {indices.dtype})")

        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 1 and value.shape[0] == self._capacity:
                if verbose:
                    print(f"Sorting {self._count:,} elements of {key}")
                sort = np.zeros_like(value)
                sort[: self._count] = value[indices]
                self.__dict__[key] = sort

        return

    def squash(self, indices, verbose: bool = False) -> None:
        """
        Reduces the active count of the internal numpy arrays keeping only elements True in the provided boolean indices.

        Parameters:

            indices (np.ndarray): A boolean array indicating which elements to keep. Must have the same length as the current population count.
            verbose (bool, optional): If True, prints detailed information about the squashing process. Defaults to False.

        Raises:

            AssertionError: If `indices` is not a boolean array or if its length does not match the current population count.

        Returns:

            None
        """

        _is_instance(indices, np.ndarray, f"Indices must be a numpy array (got {type(indices)})")
        _has_shape(indices, (self._count,), f"Indices must have the same length as the population count ({self._count})")
        _is_dtype(indices, np.bool_, f"Indices must be a boolean array (got {indices.dtype})")

        current_count = self._count
        selected_count = indices.sum()
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 1 and value.shape[0] == self._capacity:
                if verbose:
                    print(f"Squashing {key} from {current_count:,} to {selected_count:,}")
                value[:selected_count] = value[:current_count][indices]
        self._count = selected_count

        return


# Sanity checks


def _is_instance(obj, types, message):
    if not isinstance(obj, types):
        raise TypeError(message)

    return


# def _has_dimensions(obj, dimensions, message):
#     if not len(obj.shape) == dimensions:
#         raise TypeError(message)

#     return


def _is_dtype(obj, dtype, message):
    if not np.issubdtype(obj.dtype, dtype):
        raise TypeError(message)

    return


# def _has_values(check, message):
#     if not np.all(check):
#         raise ValueError(message)

#     return


def _has_shape(obj, shape, message):
    if not obj.shape == shape:
        raise TypeError(message)

    return
