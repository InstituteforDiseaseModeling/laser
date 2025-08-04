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
    count (int): The current count of active elements.
    capacity (int): The maximum capacity of the frame.
"""

import h5py
import numpy as np

from laser_core.utils import calc_capacity


class LaserFrame:
    """
    The LaserFrame class, similar to a db table or a Pandas DataFrame, holds dynamically
    allocated data for agents (generally 1-D or scalar) or for nodes|patches (e.g., 1-D for
    scalar value per patch or 2-D for time-varying per patch)."""

    def __init__(self, capacity: int, initial_count: int = -1, **kwargs):
        """
        Initialize a LaserFrame object.

        Parameters:
            capacity (int): The maximum capacity of the frame.
                            Must be a positive integer.
            initial_count (int): The initial number of active elements in the frame.
                                 Must be a positive integer <= capacity.
            **kwargs: Additional keyword arguments to set as attributes of the object.

        Raises:
            ValueError: If capacity or initial_count is not a positive integer,
                        or if initial_count is greater than capacity.

        Returns:
            None
        """
        if not isinstance(capacity, (int, np.integer)) or capacity <= 0:
            raise ValueError(f"Capacity must be a positive integer, got {capacity}.")

        if initial_count == -1:
            initial_count = capacity

        if not isinstance(initial_count, (int, np.integer)) or initial_count < 0:
            raise ValueError(f"Initial count must be a non-negative integer, got {initial_count}.")

        if initial_count > capacity:
            raise ValueError(f"Initial count ({initial_count}) cannot exceed capacity ({capacity}).")

        self._count = initial_count
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
        stored as a 1-D NumPy array (scalar / entry) with a specified data type and default value.

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

        This method initializes a new property with the given name as a 2-D NumPy array (vector per entry).

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

    def add_array_property(self, name: str, shape: tuple, dtype=np.uint32, default=0) -> None:
        """
        Adds an array property to the object.

        This method initializes a new property with the given name as a multi-dimensional NumPy array.

        The array will have the given shape (note that there is no implied dimension of size self._capacity),
        datatype (default is np.uint32), and default value (default is 0).

        Parameters:

            name (str): The name of the property to be added.
            shape (tuple): The shape of the array.
            dtype (data-type, optional): The desired data-type for the array, default is np.uint32.
            default (scalar, optional): The default value to fill the array with, default is 0.

        Returns:

            None

        """

        # initialize the property to a NumPy array with given shape, dtype, and default value
        setattr(self, name, np.full(shape, default, dtype=dtype))
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

        This method increments the internal count by the given count, ensuring that the total does not exceed the frame's capacity. If the addition would exceed the capacity, an assertion error is raised. This method is typically used to add new births during the simulation.

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

            indices (np.ndarray): An array of indices used to sort the numpy arrays. Must be of integer type and have the same length as the frame count (`self._count`).

            verbose (bool, optional): If True, prints the sorting progress for each numpy array attribute. Defaults to False.

        Raises:

            AssertionError: If `indices` is not an integer array or if its length does not match the frame count of active elements.
        """

        _is_instance(indices, np.ndarray, f"Indices must be a numpy array (got {type(indices)})")
        _has_shape(indices, (self._count,), f"Indices must have the same length as the frame active element count ({self._count})")
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

            indices (np.ndarray): A boolean array indicating which elements to keep. Must have the same length as the current frame active element count.
            verbose (bool, optional): If True, prints detailed information about the squashing process. Defaults to False.

        Raises:

            AssertionError: If `indices` is not a boolean array or if its length does not match the current frame active element count.

        Returns:

            None
        """

        _is_instance(indices, np.ndarray, f"Indices must be a numpy array (got {type(indices)})")
        _has_shape(indices, (self._count,), f"Indices must have the same length as the frame active element count ({self._count})")
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

    def save_snapshot(self, path, results_r=None, pars=None):
        """
        Save this LaserFrame and optional extras to an HDF5 snapshot file.

        Parameters:
            path: Destination file path
            results_r: Optional 2D numpy array of recovered counts
            pars: Optional PropertySet or dict of parameters
        """
        from laser_core.propertyset import PropertySet  # to avoid circular import

        with h5py.File(path, "w") as f:
            self._save(f, "people")

            if results_r is not None:
                f.create_dataset("recovered", data=results_r)

            if pars is not None and isinstance(pars, (dict, PropertySet)):
                data = pars.to_dict() if isinstance(pars, PropertySet) else pars
                self._save_dict(data, f.create_group("pars"))

    def _save(self, parent_group, name):
        """
        Internal method to save this LaserFrame under the given group name.
        """
        group = parent_group.create_group(name)
        group.attrs["count"] = self._count
        group.attrs["capacity"] = self._capacity

        for key in dir(self):
            if not key.startswith("_"):
                value = getattr(self, key)
                if isinstance(value, np.ndarray):
                    data = value[: self._count]
                    group.create_dataset(key, data=data)

    def _save_dict(self, data, group):
        """
        Internal method to save a dict as datasets and attributes in a group.
        """
        for key, value in data.items():
            try:
                group.create_dataset(key, data=value)
            except TypeError:
                group.attrs[key] = str(value)

    @classmethod
    def load_snapshot(cls, path, n_ppl, cbr, nt):
        """
        Load a LaserFrame and optional extras from an HDF5 snapshot file.

        Args:
            path (str): Path to the HDF5 snapshot file.
            n_ppl (float or array-like): Original total population (or per-node array) used to estimate births.
            cbr (float or array-like): Crude birth rate (per 1000/year).
            nt (int): Simulation duration (number of ticks).

        Returns:
            frame (LaserFrame)
            results_r (np.ndarray or None)
            pars (dict or None)
        """

        with h5py.File(path, "r") as f:
            group = f["people"]
            count = int(group.attrs["count"])

            # Load parameters first
            if "pars" in f:
                pars_group = f["pars"]
                pars = {
                    key: (pars_group[key][()].decode() if isinstance(pars_group[key][()], bytes) else pars_group[key][()])
                    for key in pars_group
                }
                pars.update({key: (val.decode() if isinstance(val, bytes) else val) for key, val in pars_group.attrs.items()})
            else:
                pars = {}

            # Compute capacity if values are provided
            if n_ppl is not None and cbr is not None and nt is not None:
                if isinstance(cbr, (list, np.ndarray)) and len(cbr) > 1:
                    cbr_value = np.sum(cbr * n_ppl) / np.sum(n_ppl)
                else:
                    cbr_value = cbr[0] if isinstance(cbr, (list, np.ndarray)) else cbr
                ppl = np.sum(n_ppl)
                expected_births = calc_capacity(ppl, nt, cbr_value) - ppl
                # Fudge factor: We multiply the expected number of births by a small safety margin
                # to ensure our estimate is higher than any likely realized (stochastic) outcome.
                # This accounts for randomness in actual births drawn from a Poisson (or similar) process.
                # The chosen multiplier (e.g., 1.025) is based on empirical trials and reflects the
                # low but non-zero probability of occasional spikes in simulated births.
                # Too large a fudge factor wastes memory; too small risks overflow. 1.025 is a balance.
                fudge_factor = 1 + 4 / np.sqrt(expected_births)
                capacity = int(fudge_factor * (count + expected_births))
            else:
                capacity = count

            if capacity < count:
                raise ValueError(f"There is no way capacity ({capacity}) should ever be less than count ({count}).")

            # Now construct frame
            frame = cls(capacity=capacity, initial_count=count)
            for key in group:
                data = group[key][:]
                dtype = data.dtype
                frame.add_scalar_property(name=key, dtype=dtype, default=0)
                getattr(frame, key)[:count] = data

            results_r = f["recovered"][()] if "recovered" in f else None

        return frame, results_r, pars


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
