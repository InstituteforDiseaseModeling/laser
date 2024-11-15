"""Utility functions for the IDMLaser package."""

from collections.abc import Iterable
from json import JSONEncoder
from pathlib import Path
from typing import Any
from typing import Tuple
from typing import Union

import geopandas as gpd
import requests
import zipfile
from PyPDF2 import PdfMerger
import glob
import os
import pdb

import numba as nb
import numpy as np
import matplotlib.pyplot as plt # for viz function
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

pdf_filename = 'combined_plots.pdf' # should probably be in manifest?

def pop_from_cbr_and_mortality(
    initial: Union[int, np.integer],
    cbr: Union[Iterable[Union[int, float, np.number]], Union[int, float, np.number]],
    mortality: Union[Iterable[Union[int, float, np.number]], Union[int, float, np.number]],
    nyears: Union[int, np.integer],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate births, deaths, and total population for each year."""
    births = np.zeros(nyears, dtype=np.uint32)
    deaths = np.zeros(nyears, dtype=np.uint32)
    population = np.zeros(nyears + 1, dtype=np.uint32)

    if not isinstance(cbr, Iterable):
        cbr = np.full(nyears, cbr)
    if not isinstance(mortality, Iterable):
        mortality = np.full(nyears, mortality)

    population[0] = initial

    for year, cbr_i, mortality_i in zip(range(nyears), cbr, mortality):
        current_population = population[year]
        births[year] = np.random.poisson(cbr_i * current_population / 1000)
        deaths[year] = np.random.poisson(mortality_i * current_population / 1000)
        population[year + 1] = population[year] + births[year] - deaths[year]

    return births, deaths, population


def daily_births_deaths_from_annual(annual_births, annual_deaths) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate daily births and deaths from annual values."""
    assert len(annual_births) == len(annual_deaths), "Lengths of arrays must be equal"
    nyears = len(annual_births)
    daily_births = np.zeros(nyears * 365, dtype=np.uint32)
    daily_deaths = np.zeros(nyears * 365, dtype=np.uint32)

    for day in range(nyears * 365):
        year = day // 365
        doy = (day % 365) + 1
        daily_births[day] = (annual_births[year] * doy // 365) - (annual_births[year] * (doy - 1) // 365)
        daily_deaths[day] = (annual_deaths[year] * doy // 365) - (annual_deaths[year] * (doy - 1) // 365)

    return daily_births, daily_deaths


class NumpyJSONEncoder(JSONEncoder):
    """Custom JSON encoder for NumPy arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return JSONEncoder.default(self, obj)


# Derived from table 1 of "National Vital Statistics Reports Volume 54, Number 14 United States Life Tables, 2003" (see README.md)
cumulative_deaths = [
    0,
    687,
    733,
    767,
    792,
    811,
    829,
    844,
    859,
    872,
    884,
    895,
    906,
    922,
    945,
    978,
    1024,
    1081,
    1149,
    1225,
    1307,
    1395,
    1489,
    1587,
    1685,
    1781,
    1876,
    1968,
    2060,
    2153,
    2248,
    2346,
    2449,
    2556,
    2669,
    2790,
    2920,
    3060,
    3212,
    3377,
    3558,
    3755,
    3967,
    4197,
    4445,
    4715,
    5007,
    5322,
    5662,
    6026,
    6416,
    6833,
    7281,
    7759,
    8272,
    8819,
    9405,
    10032,
    10707,
    11435,
    12226,
    13089,
    14030,
    15051,
    16146,
    17312,
    18552,
    19877,
    21295,
    22816,
    24445,
    26179,
    28018,
    29972,
    32058,
    34283,
    36638,
    39108,
    41696,
    44411,
    47257,
    50228,
    53306,
    56474,
    59717,
    63019,
    66336,
    69636,
    72887,
    76054,
    79102,
    81998,
    84711,
    87212,
    89481,
    91501,
    93266,
    94775,
    96036,
    97065,
    97882,
    100000,
]

__cdnp = np.array(cumulative_deaths)


def predicted_year_of_death(age_years, max_year=100):
    """
    Calculates the predicted year of death based on the given age in years.

    Parameters:
    - age_years (int): The age of the individual in years.
    - max_year (int): The maximum year to consider for calculating the predicted year of death. Default is 100.

    Returns:
    - yod (int): The predicted year of death.

    Example:
    >>> predicted_year_of_death(40, max_year=80)
    62
    """

    # e.g., max_year == 10, 884 deaths are recorded in the first 10 years
    total_deaths = __cdnp[max_year + 1]
    # account for current age, i.e., agent is already 4 years old, so 792 deaths have already occurred
    already_deceased = __cdnp[age_years]
    # this agent will be one of the deaths in (already_deceased, total_deaths] == [already_deceased+1, total_deaths+1)
    draw = np.random.randint(already_deceased + 1, total_deaths + 1)
    # find the year of death, e.g., draw == 733, searchsorted("left") will return 2, so the year of death is 1
    yod = np.searchsorted(__cdnp, draw, side="left") - 1

    return yod


def predicted_day_of_death(age_days, max_year=100):
    """
    Calculates the predicted day of death based on the given age in days and the maximum year of death.

    Parameters:
    - age_days (int): The age in days.
    - max_year (int): The maximum year of death. Defaults to 100.

    Returns:
    - dod (int): The predicted day of death.

    The function first calculates the predicted year of death based on the given age in days and the maximum year of death.
    Then, it randomly selects a day within the year of death.
    The age/date of death has to be greater than today's age.
    Finally, it calculates and returns the predicted day of death.

    Note: This function assumes that there are 365 days in a year.
    """

    yod = predicted_year_of_death(age_days // 365, max_year)

    # if the death age year is not the current age year pick any day that year
    if age_days // 365 < yod:
        # the agent will die sometime in the year of death, so we randomly select a day
        doy = np.random.randint(365)
    else:
        # the agent will die on or before next birthday
        age_doy = age_days % 365  # 0 ... 364
        if age_doy < 364:
            # there is time before the next birthday, pick a day at random
            doy = np.random.randint(age_doy + 1, 365)
        else:
            # the agent's birthday is tomorrow; bummer of a birthday present
            yod += 1
            doy = 0

    dod = yod * 365 + doy

    return dod


class PropertySet:
    def __init__(self, *bags):
        for bag in bags:
            assert isinstance(bag, (type(self), dict))
            for key, value in (bag.__dict__ if isinstance(bag, type(self)) else bag).items():
                setattr(self, key, value)

    def __add__(self, other):
        return PropertySet(self, other)

    def __iadd__(self, other):
        assert isinstance(other, (type(self), dict))
        for key, value in (other.__dict__ if isinstance(other, type(self)) else other).items():
            setattr(self, key, value)
        return self

    def __len__(self):
        return len(self.__dict__)

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return f"PropertySet({self.__dict__!s})"


class PriorityQueuePy:
    """
    A priority queue implemented using NumPy arrays and sped-up with Numba.
    Using the algorithm from the Python heapq module.
    __init__ with an existing array of priority values
    __push__ with an index into priority values
    __pop__ returns the index of the highest priority value and its value
    """

    # https://github.com/python/cpython/blob/5592399313c963c110280a7c98de974889e1d353/Modules/_heapqmodule.c
    # https://github.com/python/cpython/blob/5592399313c963c110280a7c98de974889e1d353/Lib/heapq.py

    def __init__(self, capacity: int, values: np.ndarray):
        self.indices = np.zeros(capacity, dtype=np.uint32)
        self.values = values
        self.size = 0

        return

    def push(self, index) -> None:
        if self.size >= len(self.indices):
            raise IndexError("Priority queue is full")
        self.indices[self.size] = index
        _siftdown(self.indices, self.values, 0, self.size)
        self.size += 1
        return

    def peeki(self) -> np.uint32:
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return self.indices[0]

    def peekv(self) -> Any:
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return self.values[self.indices[0]]

    def peekiv(self) -> Tuple[np.uint32, Any]:
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return (self.indices[0], self.values[self.indices[0]])

    def popi(self) -> np.uint32:
        index = self.peeki()
        self.pop()

        return index

    def popv(self) -> Any:
        value = self.peekv()
        self.pop()

        return value

    def popiv(self) -> Tuple[np.uint32, Any]:
        ivtuple = self.peekiv()
        self.pop()

        return ivtuple

    def pop(self) -> None:
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        self.size -= 1
        self.indices[0] = self.indices[self.size]
        _siftup(self.indices, self.values, 0, self.size)
        return

    def __len__(self):
        return self.size


@nb.njit((nb.uint32[:], nb.int32[:], nb.uint32, nb.uint32), nogil=True)
def _siftdown(indices, values, startpos, pos):
    inewitem = indices[pos]
    vnewitem = values[inewitem]
    # Follow the path to the root, moving parents down until finding a place newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        iparent = indices[parentpos]
        vparent = values[iparent]
        if vnewitem < vparent:
            indices[pos] = iparent
            pos = parentpos
            continue
        break
    indices[pos] = inewitem

    return


@nb.njit((nb.uint32[:], nb.int32[:], nb.uint32, nb.uint32), nogil=True)
def _siftup(indices, values, pos, size):
    endpos = size
    startpos = pos
    inewitem = indices[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2 * pos + 1  # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not values[indices[childpos]] < values[indices[rightpos]]:
            childpos = rightpos
        # Move the smaller child up.
        indices[pos] = indices[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    indices[pos] = inewitem
    _siftdown(indices, values, startpos, pos)
    return

def viz_2D( model, data, label, x_label, y_label ):
    """
    Visualize a 2D array as a heatmap using a specified color map, with options for custom labels and an automatically scaled color bar.

    Parameters
    ----------
    data : 2D array-like
        The data array to visualize. Each element in the array is represented as a pixel, with color indicating value.
        Values are displayed in their raw range without normalization, allowing for both positive and negative values.

    label : str
        Title for the heatmap, displayed above the plot.

    x_label : str
        Label for the x-axis, typically describing the column dimension or variable associated with the horizontal axis.

    y_label : str
        Label for the y-axis, typically describing the row dimension or variable associated with the vertical axis.

    Notes
    -----
    - The colormap used is 'viridis' by default, providing a perceptually uniform color gradient suitable for a wide range of data values.
    - The color bar is dynamically scaled to the actual range of `data`, so values are represented accurately, including any negatives.
    - The aspect ratio is set to 'auto' to ensure each data element appears as a distinct pixel in the visualization.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(10, 10) * 2 - 1  # Generate a 10x10 array with values between -1 and 1
    >>> viz_2D(data, label="Random Data Heatmap", x_label="X Axis", y_label="Y Axis")

    This function is ideal for visualizing small-to-medium-sized 2D datasets where each value should be represented individually.
    """

    # Set up the figure with a specific aspect ratio
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the heatmap with 1 pixel per value
    cax = ax.imshow(data, cmap='viridis', interpolation='none', aspect='auto', origin='lower')
    
    # Adjust color bar size and label
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value')
    
    plt.title(label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if model.params.to_pdf:
        with PdfPages(label.replace( " ", "_" )+".pdf") as pdf:
            pdf.savefig()
            plt.close()
    else:
        plt.show()

def load_csv_maybe_header(file_path):
    """
    Load a CSV file as a NumPy array, automatically detecting and skipping a header row if present.

    Parameters
    ----------
    file_path : str or Path-like
        Path to the CSV file to be loaded.

    Returns
    -------
    data : ndarray
        A NumPy array containing the data from the CSV file. If a header row is detected, it is skipped,
        and only the numerical data is returned.

    Notes
    -----
    This function performs a quick heuristic check on the first line of the file to determine if it is a header row.
    A row is identified as a header if any of the following conditions are met for any item in the first row:
    - Contains alphabetic characters.
    - Contains a hyphen or minus sign (indicating text or non-numeric values).
    - Has a numeric value outside the range [0, 1] (assuming all valid data falls within this range).

    If the function detects a header, it skips the first line when loading the data with `np.loadtxt`. Otherwise, 
    it reads from the start of the file.

    Examples
    --------
    >>> # Suppose 'data.csv' has a header row
    >>> load_csv_maybe_header('data.csv')
    array([[0.1, 0.2, 0.3],
           [0.4, 0.5, 0.6]])

    >>> # Suppose 'data_no_header.csv' has no header row
    >>> load_csv_maybe_header('data_no_header.csv')
    array([[0.1, 0.2, 0.3],
           [0.4, 0.5, 0.6]])

    This function is particularly useful for cases where CSV files may inconsistently include a header row.

    Raises
    ------
    ValueError
        If any item in the first line cannot be converted to a float when performing header detection checks.

    """
    # Open file to check the first line
    with open(file_path, 'r') as f:
        first_line = f.readline().strip().split(',')
        
        # Check if first line contains a header based on conditions
        has_header = any(
            item.isalpha() or  # Check if item contains letters
            '-' in item or     # Check if item has minus/hyphen
            float(item) < 0 or float(item) > 1
            for item in first_line
        )
    
    # Load data with np.loadtxt, skip header if detected
    data = np.loadtxt(file_path, delimiter=',', skiprows=1 if has_header else 0)
    return data

def viz_pop(model, url="https://packages.idmod.org:443/artifactory/idm-data/LASER/ssa_shapes.zip", extract_to_dir="ssa_shapes"):
    """
    Download, unzip, and plot population data on a map of Sub-Saharan Africa.

    Parameters:
        url (str): URL to download the zip file from.
        extract_to_dir (str or Path): Directory name where the zip file will be extracted.
    """
    # Download the file
    zip_path = Path(f"{extract_to_dir}.zip")
    if not zip_path.exists():
        print("Downloading zip file...")
        response = requests.get(url)
        response.raise_for_status()
        with open(zip_path, "wb") as file:
            file.write(response.content)
        print("Download complete.")

    # Unzip the file
    extract_to_dir = Path(extract_to_dir)
    if not extract_to_dir.exists():
        print("Unzipping file...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to_dir)
        print("Unzipping complete.")

    # Load data from synth_small_ssa module
    import synth_small_ssa as ssa
    nn_nodes, initial_populations, cbrs = ssa.run()

    # Extract longitude, latitude, and population data
    nn_longitudes = [node[1][1] for node in nn_nodes.values()]
    nn_latitudes = [node[1][0] for node in nn_nodes.values()]
    nn_populations = [node[0][0] for node in nn_nodes.values()]
    nn_sizes = 0.05 * np.sqrt(nn_populations)  # Marker sizes scaled by population

    # Load and plot shapefile
    shapefile_path = extract_to_dir / "ssa_shapes.shp"
    shapefile = gpd.read_file(shapefile_path)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)

    # Plot shapefile data
    shapefile.plot(ax=ax)

    # Plot the population data with logarithmic color scaling
    scatter = ax.scatter(
        nn_longitudes, nn_latitudes, 
        s=nn_sizes, 
        c=nn_populations, 
        norm=LogNorm(), 
        cmap="inferno"
    )

    # Add a color bar and labels
    plt.colorbar(scatter, label="Population")
    if model.params.to_pdf:
        with PdfPages("pop_plot.pdf") as pdf:
            pdf.savefig()
            plt.close()
    else:
        plt.show()

def combine_pdfs(output_filename=pdf_filename):
    print( "DEBUG: combine_pdfs" )
    merger = PdfMerger()

    pdf_files = glob.glob("*.pdf")
    # Append each individual PDF file to the merger
    pdf_files.remove( output_filename )
    for pdf in pdf_files:
        print( f"Found and appending {pdf}." )
        merger.append(pdf)

    # Write out the combined PDF
    with open(output_filename, "wb") as fout:
        merger.write(fout)

    for pdf in pdf_files:
        os.remove( pdf )

    print(f"Combined PDF saved to {output_filename}.")

# Usage
#download_unzip_plot("https://packages.idmod.org:443/artifactory/idm-data/LASER/ssa_shapes.zip")

# """
# push/pop elements/sec for various priority queue implementations
# |impl|push()|pop()|
# |----|:----:|:---:|
# |PriorityQueue  |  725,450|   82,340|
# |PriorityQueueNP|  815,173|   80,183|
# |PriorityQueueNB|1,096,055|  847,965|
# |PriorityQueuePy|1,865,557|  934,897|
# |PythonHeapQ    |5,581,031|3,202,212|
# """
