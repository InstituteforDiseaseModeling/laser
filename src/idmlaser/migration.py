import numpy as np


def set_max_outflow(network, max_frac):
    outflows = network.sum(axis=0)
    ii = np.argwhere(outflows > max_frac)
    for i in ii:
        network[i, :] *= max_frac / outflows[i]

    return network


def gravity(params, pops, distances):
    """
    Gravity model for migration between nodes

    Args:
        params (dict): Parameters for the gravity model
        pops (np.ndarray): List of populations for each node
        distances (np.ndarray): Distance matrix between nodes in km

    Returns:
        np.ndarray: Migration network between nodes (i,j) where j --> i
    """

    num_nodes = len(pops)
    network = distances.copy()
    network *= 1000  # convert to meters
    for i in range(num_nodes):
        popi = pops[i]
        for j in range(i + 1, num_nodes):
            popj = pops[j]
            network[i, j] = network[j, i] = params["k"] * (popi ** params["a"]) * (popj ** params["b"]) / (network[i, j] ** params["c"])
    network /= np.sum(pops)

    if "max_frac" in params:
        network = set_max_outflow(network, params["max_frac"])

    return network


def radiation(params, pops, distances):
    sort_indices = np.argsort(distances, axis=1)
    unsort_indices = np.argsort(sort_indices, axis=1)

    network = np.zeros_like(distances)

    for i in range(len(pops)):
        sorted_pops = pops[sort_indices[i]]
        cumulative_sorted_pops = np.cumsum(sorted_pops)

        if not params["include_home"]:
            cumulative_sorted_pops = cumulative_sorted_pops - sorted_pops[0]

        network[i] = (
            params["k"] * pops[i] * sorted_pops / (pops[i] + cumulative_sorted_pops) / (pops[i] + sorted_pops + cumulative_sorted_pops)
        )

    network = np.take_along_axis(network, unsort_indices, axis=1).T
    np.fill_diagonal(network, 0)

    if "max_frac" in params:
        network = set_max_outflow(network, params["max_frac"])

    return network


def stouffer(params, pops, distances):
    sort_indices = np.argsort(distances, axis=1)
    unsort_indices = np.argsort(sort_indices, axis=1)

    network = np.zeros_like(distances)
    for i in range(len(pops)):
        sorted_pops = pops[sort_indices[i]]

        # could be handled more
        cumulative_sorted_pops = np.cumsum(sorted_pops)
        if not params["include_home"]:
            cumulative_sorted_pops = cumulative_sorted_pops - sorted_pops[0]

        network[i][1:] = params["k"] * pops[i] ** params["a"] * (sorted_pops[1:] / cumulative_sorted_pops[1:]) ** params["b"]

    network = np.take_along_axis(network, unsort_indices, axis=1).T
    np.fill_diagonal(network, 0)

    if "max_frac" in params:
        network = set_max_outflow(network, params["max_frac"])

    return network


def competing_destinations(params, pops, distances):
    network = gravity({k: params[k] for k in ["a", "c", "b", "k"]}, pops, distances)
    interference_matrix = pops ** params["b"] * distances ** (-1 * params["c"])
    # I suppose reusing the gravity model means we do some computations twice, but this seems a small thing.

    row_sums = np.sum(interference_matrix, axis=1) - np.diag(interference_matrix)
    # Rather than computing that interior sum for each element, just compute the row sums,
    # and then subtract the excluded elements
    for i in range(len(pops)):
        for j in range(len(pops)):
            if j != i:
                network[i][j] = network[i][j] * (row_sums[i] - interference_matrix[i][j]) ** params["delta"]

    network = network.T
    np.fill_diagonal(network, 0)

    if "max_frac" in params:
        network = set_max_outflow(network, params["max_frac"])

    return network
