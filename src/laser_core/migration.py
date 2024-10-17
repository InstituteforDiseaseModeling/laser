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

import numpy as np


def shared_sanity_checks(pops, distances, **params):
    _is_instance(pops, np.ndarray, f"pops must be a NumPy array ({type(pops)=})")
    _has_dimensions(pops, 1, f"pops must be a 1D array ({pops.shape=})")
    _is_dtype(pops, np.number, f"pops must be a numeric array ({pops.dtype=})")
    _has_values(pops >= 0, "pops must contain only non-negative values")

    _is_instance(distances, np.ndarray, f"distances must be a NumPy array ({type(distances)=})")
    _has_dimensions(distances, 2, f"distances must be a 2D array ({distances.shape=})")
    _has_shape(
        distances,
        (pops.shape[0], pops.shape[0]),
        f"distances must be a square matrix with length equal to the length of pops ({distances.shape=}, {pops.shape=})",
    )
    _is_dtype(distances, np.number, f"distances must be a numeric array ({distances.dtype=})")
    _has_values(distances >= 0, "distances must contain only non-negative values")
    _has_values(distances == distances.T, "distances must be a symmetric matrix")

    a = params.get("a", None)
    if a is not None:
        _is_instance(a, Number, f"a must be a numeric value ({type(a)=})")
        _has_values(a >= 0, f"a must be a non-negative value ({a=})")

    b = params.get("b", None)
    if b is not None:
        _is_instance(b, Number, f"b must be a numeric value ({type(b)=})")
        _has_values(b >= 0, f"b must be a non-negative value ({b=})")

    c = params.get("c", None)
    if c is not None:
        _is_instance(c, Number, f"c must be a numeric value ({type(c)=})")
        _has_values(c >= 0, f"c must be a non-negative value ({c=})")

    delta = params.get("delta", None)
    if delta is not None:
        _is_instance(delta, Number, f"delta must be a numeric value ({type(delta)=})")

    k = params.get("k", None)
    if k is not None:
        _is_instance(k, Number, f"k must be a numeric value ({type(k)=})")
        _has_values(k >= 0, f"k must be a non-negative value ({k=})")

    include_home = params.get("include_home", None)
    if include_home is not None:
        _is_instance(include_home, (int, bool), f"include_home must be boolean or integer type ({type(include_home)=})")


def gravity(pops: np.ndarray, distances: np.ndarray, k: float, a: float, b: float, c: float, **kwargs):
    """
    Calculate a gravity model network with an optional maximum export fraction constraint.
    This function computes a gravity model network based on the provided populations
    and distances, and, if specified, then normalizes the rows of the resulting network matrix
    such that no row exceeds the specified maximum export fraction.
    Mathematical formula:
        element-by-element: network_{i,j} = k * p_i^a * p_j^b / distance_{i,j}^c
        as implemented in numpy math: network = k * (pops[:, np.newaxis] ** a) * (pops ** b) * (distances ** (-1 * c))

    Parameters:
    pops (numpy.ndarray): 1D array of populations.
    distances (numpy.ndarray): 2D array of distances between the populations.
    k (float): Scaling constant.
    a (float): Exponent for the population of the origin.
    b (float): Exponent for the population of the destination.
    c (float): Exponent for the distance.
    **kwargs: Additional keyword arguments (not used in the current implementation).
    Returns:
    numpy.ndarray: A matrix representing the gravity model network with rows optionally
    normalized to respect the maximum export fraction.
    """
    # KM: Taking "max_frac" as a parameter.  The "row_normalizer" can be applied to any network, so let's just make it a separate helper function
    # without tying it to the gravity model (or adding max_frac as a parameter to every network, where all it does is pass the final network through row_normalizer
    #  - if we want to make a migration model class instead of separate functions, we can revisit that)
    # Sanity checks
    shared_sanity_checks(pops, distances, a=a, b=b, c=c, k=k)

    distances1 = distances.copy()
    np.fill_diagonal(distances1, 1)  # Prevent division by zero in `distances ** (-1 * c)`
    network = k * (pops[:, np.newaxis] ** a) * (pops**b) * (distances1 ** (-1 * c))

    np.fill_diagonal(network, 0)

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


def competing_destinations(pops, distances, k, a, b, c, delta, **params):
    """
    Calculate the competing destinations model for a given set of populations and distances. (Fotheringham AS. Spatial flows and spatial patterns. Environment and planning A. 1984;16(4):529–543)
    This function computes a network matrix based on the gravity model and then adjusts it
    using the competing destinations model. The adjustment is done by considering the
    interference from other destinations.
    Mathematical formula:
        element-by-element: network_{i,j} = k * p_i^a * p_j^b / distance_{i,j}^c * (sum_k (p_k^b / distance_{j,k}^c) for k not in [i,j] )^delta ))
        as-implemented numpy math:
        compute all terms up to the sum_k using the gravity model
        Construct the matrix inside the sum: p**b * distances **(1-c)
        Sum on the second axis (k), and subtract off the diagonal (j=k terms): row_sums = np.sum(competition_matrix, axis=1) - np.diag(competition_matrix)
        Now element-by-element, subtract k=i terms off the sum, exponentiate, and multiple the original network term:
            network[i][j] = network[i][j] * (row_sums[i] - competition_matrix[i][j]) ** delta

    Parameters:
    pops (numpy.ndarray): Array of populations.
    distances (numpy.ndarray): Array of distances between locations.
    k (float): Scaling constant.
    a (float): Exponent for the population of the origin.
    b (float): Exponent parameter for populations in the gravity model.
    c (float): Exponent parameter for distances in the gravity model.
    delta (float): Exponent parameter for the competing destinations adjustment.
    **params: Additional parameters to be passed to the gravity model.
    Returns:
    numpy.ndarray: Adjusted network matrix based on the competing destinations model.
    """

    # Sanity checks
    shared_sanity_checks(pops, distances, a=a, k=k, b=b, c=c, delta=delta)

    network = gravity(pops, distances, k=k, a=a, b=b, c=c, **params)
    # Construct the p_j^b / d_jk^c matrix, inside the sum
    distances1 = distances.copy()
    np.fill_diagonal(distances1, 1)  # Prevent division by zero in `distances ** (-1 * c)`
    competition_matrix = pops**b * distances1 ** (-1 * c)

    # Sum over all k, and remove the terms j=k
    # Don't subtract off the diagonal - this could be a NaN if distance to self is zero
    np.fill_diagonal(competition_matrix, 0)
    mysums = np.sum(competition_matrix, axis=1)
    # Rather than computing that interior sum for each element, just compute the row sums,
    # and then subtract the k=i term.
    for i in range(len(pops)):
        for j in range(len(pops)):
            if j != i:
                network[i][j] = network[i][j] * (mysums[j] - competition_matrix[j][i]) ** delta

    np.fill_diagonal(network, 0)
    return network


def sum_populations_as_close_or_closer(sorted_pops, sorted_distance_row):
    # Separating this operation out because it is common between a couple of migration models, and there is a little trickiness in appropriately
    # handling cases where there are multiple destinations equidistant from the source node.
    _is_instance(sorted_pops, np.ndarray, f"sorted_pops must be a NumPy array ({type(sorted_pops)=})")
    _is_dtype(sorted_pops, np.number, f"sorted_pops must be a numeric array ({sorted_pops.dtype=})")
    _has_values(sorted_pops >= 0, "sorted_pops must contain only non-negative values")
    _is_instance(sorted_distance_row, np.ndarray, f"sorted_distance_row must be a NumPy array ({type(sorted_distance_row)=})")
    _is_dtype(sorted_distance_row, np.number, f"sorted_distance_row must be a numeric array ({sorted_distance_row.dtype=})")
    _has_values(sorted_distance_row >= 0, "sorted_distance_row must contain only non-negative values")
    _has_shape(
        sorted_pops,
        sorted_distance_row.shape,
        f"sorted_pops & sorted_distance_row must have same shape ({sorted_pops.shape=}, {sorted_distance_row.shape=})",
    )
    _has_values(np.diff(sorted_distance_row) >= 0, "sorted_distance_row must be sorted in ascending order")

    # if all distances from the source node are unique, this is easy
    cumulative_sorted_pops = np.cumsum(sorted_pops)

    if len(np.unique(sorted_distance_row)) < len(sorted_distance_row):
        # However, if there are non-unique distances, then we have to account for each equidistant node in the sum,
        # as these models tend to sum over "all nodes k such that d_ik <= d_ij".
        # Since the vector is sorted, this just means to find the first and last occurences of occurence of each unique distance,
        # and replace the cumulative sum of populations for all of those terms with the last one.
        _, start_indices = np.unique(sorted_distance_row, return_index=True)
        _, end_indices = np.unique(sorted_distance_row[::-1], return_index=True)
        end_indices = len(sorted_distance_row) - end_indices - 1

        for idx in range(len(start_indices)):
            cumulative_sorted_pops[start_indices[idx] : end_indices[idx]] = cumulative_sorted_pops[end_indices[idx]]

    return cumulative_sorted_pops


def stouffer(pops, distances, k, a, b, include_home, **params):
    """
    Computes a migration network using a modified Stouffer's model.  (Stouffer SA. Intervening opportunities: a theory relating mobility and distance. American sociological review. 1940;5(6):845–867)
    Mathematical formula:
        element-by-element: network_{i,j} = k * p_i * p_j / ( (p_i + sum_k p_k) (p_i + p_j + sum_k p_k) )
        the parameter include_home determines whether p_i is included or excluded from the sum
        as-implemented numpy math:
        Sort each row of the distance matrix (I'll use ' below to indicate distance-sorted vectors)
        Loop over "source nodes" i:
            Cumulative sum the sorted populations, ensuring appropriate handling when there are multiple destinations equidistant from the source
            Subtract the source node population if include_home is false
            Construct the row of the network matrix as k * p_i^a * (p_j' / sum_k' p_k')^b
        Unsort the rows of the network
    Parameters:
    pops (numpy.ndarray): An array of population sizes.
    distances (numpy.ndarray): A 2D array where distances[i][j] is the distance from location i to location j.
    k (float): A scaling factor for the migration rates.
    a (float): Exponent applied to the population size of the origin.
    b (float): Exponent applied to the ratio of destination population to the sum of all populations at equal or lesser distances.
    include_home (bool): If True, includes the home population in the cumulative sum; otherwise, excludes it.
    **params: Additional parameters (not used in the current implementation).
    Returns:
    numpy.ndarray: A 2D array representing the migration network, where network[i][j] is the migration rate from location i to location j.
    """

    # Sanity checks
    shared_sanity_checks(pops, distances, a=a, b=b, k=k, include_home=include_home)

    # We will just use the "truthiness" of include_home (could be boolean, could be 0/1)

    network = np.zeros_like(distances)
    sort_indices = np.argsort(distances, axis=1, kind="stable")
    unsort_indices = np.argsort(sort_indices, axis=1)

    for i in range(len(pops)):
        sorted_pops = pops[sort_indices[i]]
        cumulative_sorted_pops = sum_populations_as_close_or_closer(sorted_pops, distances[i][sort_indices[i]])

        if not include_home:
            cumulative_sorted_pops = cumulative_sorted_pops - sorted_pops[0]

        network[i, 1:] = k * pops[i] ** a * (sorted_pops[1:] / cumulative_sorted_pops[1:]) ** b
    network = np.take_along_axis(network, unsort_indices, axis=1)

    np.fill_diagonal(network, 0)
    return network


def radiation(pops, distances, k, include_home, **params):
    """
    Calculate the migration network using the radiation model.  (Simini F, Gonza ́lez MC, Maritan A, Baraba ́si AL. A universal model for mobility and migration patterns. Nature. 2012;484(7392):96–100.)
    Mathematical formula:
        element-by-element: network_{i,j} = k * p_i^a * (p_j / sum_k (p_k) )^b, where the sum proceeds over all k such that distances_{i,k} <= distances_{i,j}
        the parameter include_home determines whether p_i is included or excluded from the sum
        as-implemented numpy math:
        Sort each row of the distance matrix (I'll use ' below to indicate distance-sorted vectors)
        Loop over "source nodes" i:
            Cumulative sum the sorted populations, ensuring appropriate handling when there are multiple destinations equidistant from the source
            Subtract the source node population if include_home is false
            Construct the row of the network matrix as k * p_i * p_j' / (p_i + sum_k' p_k') / (p_i + p_j' + sum_k' p_k')
        Unsort the rows of the network
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
    shared_sanity_checks(pops, distances, k=k, include_home=include_home)

    # We will just use the "truthiness" of include_home (could be boolean, could be 0/1)

    network = np.zeros_like(distances)
    sort_indices = np.argsort(distances, axis=1, stable=True)
    unsort_indices = np.argsort(sort_indices, axis=1)

    for i in range(len(pops)):
        sorted_pops = pops[sort_indices[i]]
        cumulative_sorted_pops = sum_populations_as_close_or_closer(sorted_pops, distances[i][sort_indices[i]])

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
