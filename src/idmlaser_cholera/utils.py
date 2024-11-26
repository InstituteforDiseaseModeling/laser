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

def viz_2D( model, data, label, x_label, y_label ):
    """Visualize a 2D array as a heatmap using a specified color map, with
    options for custom labels and an automatically scaled color bar.

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
    """Load a CSV file as a NumPy array, automatically detecting and skipping a
    header row if present.

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
    """Download, unzip, and plot population data on a map of Sub-Saharan
    Africa.

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
