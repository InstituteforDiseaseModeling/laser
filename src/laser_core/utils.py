"""
This module provides utility functions for the laser-measles project.

Functions:
    calc_distances(latitudes: np.ndarray, longitudes: np.ndarray, verbose: bool = False) -> np.ndarray:
        Calculate the pairwise distances between points given their latitudes and longitudes.

    calc_capacity(population: np.uint32, nticks: np.uint32, cbr: np.float32, verbose: bool = False) -> np.uint32:
        Calculate the population capacity after a given number of ticks based on a constant birth rate.

"""

import click
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from laser_core.migration import distance


def __deprecated(msg):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            click.echo(f"WARNING: {msg}")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@__deprecated(
    "This function is deprecated and will be removed in a future release. Use the distance function from the migration module instead."
)
def calc_distances(latitudes: np.ndarray, longitudes: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Calculate the pairwise distances between points given their latitudes and longitudes.

    Parameters:

        latitudes (np.ndarray): A 1-dimensional array of latitudes.
        longitudes (np.ndarray): A 1-dimensional array of longitudes with the same shape as latitudes.
        verbose (bool, optional): If True, prints the upper left corner of the distance matrix. Default is False.

    Returns:

        np.ndarray: A 2-dimensional array where the element at [i, j] represents the distance between the i-th and j-th points.

    Raises:

        AssertionError: If latitudes is not 1-dimensional or if latitudes and longitudes do not have the same shape.
    """

    assert latitudes.ndim == 1, "Latitude array must be one-dimensional"
    assert longitudes.shape == latitudes.shape, "Latitude and longitude arrays must have the same shape"
    npatches = len(latitudes)
    distances = np.zeros((npatches, npatches), dtype=np.float32)
    for i, (lat, long) in enumerate(zip(latitudes, longitudes)):
        distances[i, :] = distance(lat, long, latitudes, longitudes)

    if verbose:
        click.echo(f"Upper left corner of distance matrix:\n{distances[0:4, 0:4]}")

    return distances


def calc_capacity(population: np.uint32, nticks: np.uint32, cbr: np.float32, verbose: bool = False) -> np.uint32:
    """
    Calculate the population capacity after a given number of ticks based on a constant birth rate (CBR).

    Args:

        population (np.uint32): The initial population.
        nticks (np.uint32): The number of ticks (time steps) to simulate.
        cbr (np.float32): The constant birth rate per 1000 people per year.
        verbose (bool, optional): If True, prints detailed population growth information. Defaults to False.

    Returns:

        np.uint32: The estimated population capacity after the given number of ticks.
    """

    # We assume a constant birth rate (CBR) for the population growth
    # The formula is: P(t) = P(0) * (1 + CBR)^t
    # where P(t) is the population at time t, P(0) is the initial population, and t is the number of ticks
    # We need to allocate space for the population data for each tick
    # We will use the maximum population growth to estimate the capacity
    daily_rate = (cbr / 1000) / 365.0  # CBR is per 1000 people per year
    capacity = np.uint32(population * (1 + daily_rate) ** nticks)

    if verbose:
        click.echo(f"Population growth: {population:,} … {capacity:,}")
        alternate = np.uint32(population * (1 + cbr / 1000) ** (nticks / 365))
        click.echo(f"Alternate growth:  {population:,} … {alternate:,}")

    return capacity


def grid(M=5, N=5, node_size_km=10, population_fn=None, origin_x=0, origin_y=0):
    """
    Create an MxN grid of cells anchored at (lat, long) with populations and geometries.

    Args:
        M (int): Number of rows (north-south).
        N (int): Number of columns (east-west).
        node_size_km (float): Size of each cell in kilometers (default 10).
        population_fn (callable): Function(row, col) returning population for a cell. Default is uniform random between 1,000 and 100,000.
        origin_x (float): longitude of the origin in decimal degrees (bottom-left corner) -180 <= origin_x < 180.
        origin_y (float): latitude of the origin in decimal degrees (bottom-left corner) -90 <= origin_y < 90.

    Returns:
        GeoDataFrame: Columns are nodeid, population, geometry.
    """

    if M < 1:
        raise ValueError("M must be >= 1")
    if N < 1:
        raise ValueError("N must be >= 1")
    if node_size_km <= 0:
        raise ValueError("node_size_km must be > 0")
    if not (-180 <= origin_x < 180):
        raise ValueError("origin_x must be -180 <= origin_x < 180")
    if not (-90 <= origin_y < 90):
        raise ValueError("origin_y must be -90 <= origin_y < 90")

    if population_fn is None:

        def population_fn(row: int, col: int) -> int:
            return int(np.random.uniform(1_000, 100_000))

    # Convert node_size_km from kilometers to degrees (approximate)
    km_per_degree = 111.320
    node_size_deg = node_size_km / km_per_degree

    cells = []
    nodeid = 0
    for row in range(M):
        for col in range(N):
            # TODO - use latitude sensitive conversion of km to degrees
            x0 = origin_x + col * node_size_deg
            y0 = origin_y + row * node_size_deg
            x1 = x0 + node_size_deg
            y1 = y0 + node_size_deg
            poly = Polygon(
                [
                    (x0, y0),  # SW
                    (x1, y0),  # SE
                    (x1, y1),  # NE
                    (x0, y1),  # NW
                    (x0, y0),  # Close polygon in SW
                ]
            )
            population = population_fn(row, col)
            if population < 0:
                raise ValueError(f"population_fn returned negative population {population} for row {row}, col {col}")
            cells.append({"nodeid": nodeid, "population": population, "geometry": poly})
            nodeid += 1

    gdf = gpd.GeoDataFrame(cells, columns=["nodeid", "population", "geometry"], crs="EPSG:4326")

    return gdf
