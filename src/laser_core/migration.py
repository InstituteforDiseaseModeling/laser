import numpy as np


def gravity(pops, distances, k, a, b, c, max_frac=None, **kwargs):
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
