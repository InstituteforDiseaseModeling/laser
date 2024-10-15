"""
This module provides various functions to calculate migration networks based on different models,
including the gravity model, competing destinations model, Stouffer's model, and the radiation model.
Additionally, it includes a utility function to calculate the great-circle distance between two points
on the Earth's surface using the Haversine formula.

Functions:
    gravity(pops: np.ndarray, distances: np.ndarray, k: float, a: float, b: float, c: float, max_frac: Union[float, None]=None, **kwargs) -> np.ndarray:

    row_normalizer(network: np.ndarray, max_rowsum: float) -> np.ndarray:
        Normalize the rows of a given network matrix such that no row sum exceeds a specified maximum value.

    competing_destinations(pops: np.ndarray, distances: np.ndarray, b: float, c: float, delta: float, **params) -> np.ndarray:

    stouffer(pops: np.ndarray, distances: np.ndarray, k: float, a: float, b: float, include_home: bool, **params) -> np.ndarray:
        Compute a migration network using a modified Stouffer's model.

    radiation(pops: np.ndarray, distances: np.ndarray, k: float, include_home: bool, **params) -> np.ndarray:

    distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        Calculate the great-circle distance between two points on the Earth's surface using the Haversine formula.
"""

from numbers import Number
from typing import Union

import numpy as np


def gravity(pops: np.ndarray, distances: np.ndarray, k: float, a: float, b: float, c: float, max_frac: Union[float, None] = None, **kwargs):
    """
    Calculate a gravity model network with an optional maximum export fraction constraint.
    This function computes a gravity model network based on the provided populations
    and distances, and, if specified, then normalizes the rows of the resulting network matrix
    such that no row exceeds the specified maximum export fraction.
    The formula used is: network = k * (pops[:, np.newaxis] ** a) * (pops ** b) * (distances ** (-1 * c))
    Parameters:
    pops (numpy.ndarray): 1D array of populations.
    distances (numpy.ndarray): 2D array of distances between the populations.
    k (float): Scaling constant.
    a (float): Exponent for the population of the origin.
    b (float): Exponent for the population of the destination.
    c (float): Exponent for the distance.
    max_frac (float, optional): The maximum fraction of any population that can be exported.
    **kwargs: Additional keyword arguments (not used in the current implementation).
    Returns:
    numpy.ndarray: A matrix representing the gravity model network with rows optionally
    normalized to respect the maximum export fraction.
    """

    # Sanity checks
    _is_instance(pops, np.ndarray, f"pops must be a NumPy array ({type(pops)=})")
    _has_dimensions(pops, 1, f"pops must be a 1D array ({pops.shape=})")
    _is_dtype(pops, np.number, f"pops must be a numeric array ({pops.dtype=})")
    _has_values(pops >= 0, "pops must contain only non-negative values")

    _is_instance(distances, np.ndarray, f"distances must be a NumPy array ({type(distances)=})")
    _has_dimensions(distances, 2, f"distances must be a 2D array ({distances.shape=})")
    _has_shape(distances, (pops.shape[0], pops.shape[0]), f"distances must be a square matrix ({distances.shape=})")
    _is_dtype(distances, np.number, f"distances must be a numeric array ({distances.dtype=})")
    _has_values(distances >= 0, "distances must contain only non-negative values")
    _has_values(distances == distances.T, "distances must be a symmetric matrix")

    _is_instance(k, Number, f"k must be a numeric value ({type(k)=})")
    _has_values(k >= 0, f"k must be a non-negative value ({k=})")

    _is_instance(a, Number, f"a must be a numeric value ({type(a)=})")
    _has_values(a >= 0, f"a must be a non-negative value ({a=})")

    _is_instance(b, Number, f"b must be a numeric value ({type(b)=})")
    _has_values(b >= 0, f"b must be a non-negative value ({b=})")

    _is_instance(c, Number, f"c must be a numeric value ({type(c)=})")
    _has_values(c >= 0, f"c must be a non-negative value ({c=})")

    if max_frac is not None:
        _is_instance(max_frac, Number, f"max_frac must be a numeric value ({type(max_frac)=})")
        _has_values(0 <= max_frac <= 1, f"max_frac must be in [0, 1] ({max_frac=})")

    distances1 = distances.copy()
    np.fill_diagonal(distances1, 1)  # Prevent division by zero in `distances ** (-1 * c)`
    network = k * (pops[:, np.newaxis] ** a) * (pops**b) * (distances1 ** (-1 * c))

    np.fill_diagonal(network, 0)

    if max_frac is not None:
        network = row_normalizer(network, max_frac)

    return network


def row_normalizer(network, max_rowsum):
    """
    Normalizes the rows of a given network matrix such that no row sum exceeds a specified maximum value.
    Parameters:
    network (numpy.ndarray): A 2D array representing the network matrix.
    max_rowsum (float): The maximum allowable sum for any row in the network matrix.
    Returns:
    numpy.ndarray: The normalized network matrix where no row sum exceeds the specified maximum value.
    """

    # Sanity checks
    _is_instance(network, np.ndarray, f"network must be a NumPy array ({type(network)=})")
    _is_dtype(network, np.number, f"network must be a numeric array ({network.dtype=})")
    _has_dimensions(network, 2, f"network must be a 2D array ({network.shape=})")
    _has_shape(network, (network.shape[0], network.shape[0]), f"network must be a square matrix ({network.shape=})")
    _has_values(network >= 0, "network must contain only non-negative values")

    _is_instance(max_rowsum, Number, f"max_rowsum must be a numeric value ({type(max_rowsum)=})")
    _has_values(0 <= max_rowsum <= 1, "max_rowsum must be in [0, 1]")

    rowsums = network.sum(axis=1)
    rows_to_renorm = rowsums > max_rowsum
    network[rows_to_renorm] = network[rows_to_renorm] * max_rowsum / rowsums[rows_to_renorm, np.newaxis]

    return network


def competing_destinations(pops, distances, b, c, delta, **params):
    """
    Calculate the competing destinations model for a given set of populations and distances.
    This function computes a network matrix based on the gravity model and then adjusts it
    using the competing destinations model. The adjustment is done by considering the
    interference from other destinations.
    Parameters:
    pops (numpy.ndarray): Array of populations.
    distances (numpy.ndarray): Array of distances between locations.
    b (float): Exponent parameter for populations in the gravity model.
    c (float): Exponent parameter for distances in the gravity model.
    delta (float): Exponent parameter for the competing destinations adjustment.
    **params: Additional parameters to be passed to the gravity model.
    Returns:
    numpy.ndarray: Adjusted network matrix based on the competing destinations model.
    """

    # Sanity checks
    _is_instance(pops, np.ndarray, f"pops must be a NumPy array ({type(pops)=})")
    _has_dimensions(pops, 1, f"pops must be a 1D array ({pops.shape=})")
    _is_dtype(pops, np.number, f"pops must be a numeric array ({pops.dtype=})")
    _has_values(pops >= 0, "pops must contain only non-negative values")

    _is_instance(distances, np.ndarray, f"distances must be a NumPy array ({type(distances)=})")
    _has_dimensions(distances, 2, f"distances must be a 2D array ({distances.shape=})")
    _has_shape(distances, (pops.shape[0], pops.shape[0]), f"distances must be a square matrix ({distances.shape=})")
    _is_dtype(distances, np.number, f"distances must be a numeric array ({distances.dtype=})")
    _has_values(distances >= 0, "distances must contain only non-negative values")

    _is_instance(b, Number, f"b must be a numeric value ({type(b)=})")
    _has_values(b >= 0, f"b must be a non-negative value ({b=})")

    _is_instance(c, Number, f"c must be a numeric value ({type(c)=})")
    _has_values(c >= 0, f"c must be a non-negative value ({c=})")

    _is_instance(delta, Number, f"delta must be a numeric value ({type(delta)=})")

    network = gravity(pops, distances, b=b, c=c, **params)
    interference_matrix = pops**b * distances ** (-1 * c)
    # I suppose reusing the gravity model means we do some computations twice, but this seems a small thing.

    row_sums = np.sum(interference_matrix, axis=1) - np.diag(interference_matrix)
    # Rather than computing that interior sum for each element, just compute the row sums,
    # and then subtract the excluded elements
    for i in range(len(pops)):
        for j in range(len(pops)):
            if j != i:
                network[i][j] = network[i][j] * (row_sums[i] - interference_matrix[i][j]) ** delta

    np.fill_diagonal(network, 0)
    return network


def stouffer(pops, distances, k, a, b, include_home, **params):
    """
    Computes a migration network using a modified Stouffer's model.
    Parameters:
    pops (numpy.ndarray): An array of population sizes.
    distances (numpy.ndarray): A 2D array where distances[i][j] is the distance from location i to location j.
    k (float): A scaling factor for the migration rates.
    a (float): Exponent applied to the population size of the origin.
    b (float): Exponent applied to the ratio of destination population to cumulative population.
    include_home (bool): If True, includes the home population in the cumulative sum; otherwise, excludes it.
    **params: Additional parameters (not used in the current implementation).
    Returns:
    numpy.ndarray: A 2D array representing the migration network, where network[i][j] is the migration rate from location i to location j.
    """

    # Sanity checks
    _is_instance(pops, np.ndarray, f"pops must be a NumPy array ({type(pops)=})")
    _has_dimensions(pops, 1, f"pops must be a 1D array ({pops.shape=})")
    _is_dtype(pops, np.number, f"pops must be a numeric array ({pops.dtype=})")
    _has_values(pops >= 0, "pops must contain only non-negative values")

    _is_instance(distances, np.ndarray, f"distances must be a NumPy array ({type(distances)=})")
    _has_dimensions(distances, 2, f"distances must be a 2D array ({distances.shape=})")
    _has_shape(distances, (pops.shape[0], pops.shape[0]), f"distances must be a square matrix ({distances.shape=})")
    _is_dtype(distances, np.number, f"distances must be a numeric array ({distances.dtype=})")
    _has_values(distances >= 0, "distances must contain only non-negative values")

    _is_instance(k, Number, f"k must be a numeric value ({type(k)=})")
    _has_values(k >= 0, f"k must be a non-negative value ({k=})")

    _is_instance(a, Number, f"a must be a numeric value ({type(a)=})")
    _has_values(a >= 0, f"a must be a non-negative value ({a=})")

    _is_instance(b, Number, f"b must be a numeric value ({type(b)=})")
    _has_values(b >= 0, f"b must be a non-negative value ({b=})")

    # We will just use the "truthiness" of include_home (could be boolean, could be 0/1)

    sort_indices = np.argsort(distances, axis=1)
    unsort_indices = np.argsort(sort_indices, axis=1)

    network = np.zeros_like(distances)
    for i in range(len(pops)):
        sorted_pops = pops[sort_indices[i]]

        cumulative_sorted_pops = np.cumsum(sorted_pops)
        if not include_home:
            cumulative_sorted_pops = cumulative_sorted_pops - sorted_pops[0]

        network[i][1:] = k * pops[i] ** a * (sorted_pops[1:] / cumulative_sorted_pops[1:]) ** b

    network = np.take_along_axis(network, unsort_indices, axis=1)
    np.fill_diagonal(network, 0)
    return network


def radiation(pops, distances, k, include_home, **params):
    """
    Calculate the migration network using the radiation model.
    Parameters:
    pops (numpy.ndarray): Array of population sizes for each node.
    distances (numpy.ndarray): 2D array of distances between nodes.
    k (float): Scaling factor for the migration rates.
    include_home (bool): Whether to include the home population in the calculations.
    **params: Additional parameters (currently not used).
    Returns:
    numpy.ndarray: 2D array representing the migration network.
    """

    # Sanity checks
    _is_instance(pops, np.ndarray, f"pops must be a NumPy array ({type(pops)=})")
    _has_dimensions(pops, 1, f"pops must be a 1D array ({pops.shape=})")
    _is_dtype(pops, np.number, f"pops must be a numeric array ({pops.dtype=})")
    _has_values(pops >= 0, "pops must contain only non-negative values")

    _is_instance(distances, np.ndarray, f"distances must be a NumPy array ({type(distances)=})")
    _has_dimensions(distances, 2, f"distances must be a 2D array ({distances.shape=})")
    _has_shape(distances, (pops.shape[0], pops.shape[0]), f"distances must be a square matrix ({distances.shape=})")
    _is_dtype(distances, np.number, f"distances must be a numeric array ({distances.dtype=})")
    _has_values(distances >= 0, "distances must contain only non-negative values")

    _is_instance(k, Number, f"k must be a numeric value ({type(k)=})")
    _has_values(k >= 0, f"k must be a non-negative value ({k=})")

    # We will just use the "truthiness" of include_home (could be boolean, could be 0/1)

    sort_indices = np.argsort(distances, axis=1)
    unsort_indices = np.argsort(sort_indices, axis=1)

    network = np.zeros_like(distances)

    for i in range(len(pops)):
        sorted_pops = pops[sort_indices[i]]
        cumulative_sorted_pops = np.cumsum(sorted_pops)

        if not include_home:
            cumulative_sorted_pops = cumulative_sorted_pops - sorted_pops[0]

        network[i] = k * pops[i] * sorted_pops / (pops[i] + cumulative_sorted_pops) / (pops[i] + sorted_pops + cumulative_sorted_pops)

    network = np.take_along_axis(network, unsort_indices, axis=1)
    np.fill_diagonal(network, 0)
    return network


def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    This function uses the Haversine formula to compute the distance between two points
    specified by their latitude and longitude in decimal degrees.
    Parameters:
    lat1 (float): Latitude of the first point in decimal degrees.
    lon1 (float): Longitude of the first point in decimal degrees.
    lat2 (float): Latitude of the second point in decimal degrees.
    lon2 (float): Longitude of the second point in decimal degrees.
    Returns:
    float: The distance between the two points in kilometers.
    """

    # Sanity checks
    _is_instance(lat1, (Number, np.ndarray), "lat1 must be a numeric value or NumPy array")
    _is_instance(lon1, (Number, np.ndarray), "lon1 must be a numeric value or NumPy array")
    _is_instance(lat2, (Number, np.ndarray), "lat2 must be a numeric value or NumPy array")
    _is_instance(lon2, (Number, np.ndarray), "lon2 must be a numeric value or NumPy array")
    _has_values((-90 <= lat1) & (lat1 <= 90), "lat1 must be in the range [-90, 90]")
    _has_values((-180 <= lon1) & (lon1 <= 180), "lon1 must be in the range [-180, 180]")
    _has_values((-90 <= lat2) & (lat2 <= 90), "lat2 must be in the range [-90, 90]")
    _has_values((-180 <= lon2) & (lon2 <= 180), "lon2 must be in the range [-180, 180]")

    # convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # haversine formula (https://en.wikipedia.org/wiki/Haversine_formula)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    RE = 6371.0  # Earth radius in km
    d = RE * c

    return d


# Sanity checks


def _is_instance(obj, types, message):
    if not isinstance(obj, types):
        raise TypeError(message)

    return


def _has_dimensions(obj, dimensions, message):
    if not len(obj.shape) == dimensions:
        raise TypeError(message)

    return


def _is_dtype(obj, dtype, message):
    if not np.issubdtype(obj.dtype, dtype):
        raise TypeError(message)

    return


def _has_values(check, message):
    if not np.all(check):
        raise ValueError(message)

    return


def _has_shape(obj, shape, message):
    if not obj.shape == shape:
        raise TypeError(message)

    return
