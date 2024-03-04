import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import sys
sys.path.append( "." )
import settings

# Read the CSV file
df = pd.read_csv(sys.argv[1])

# Function to fit a line (linear regression)
def linear_fit(x, m, b):
    return m * x + b

# Group data by 'node' and fit a line for each group
fits = {}
for node, group in df.groupby('node'):
    x_data = group['t'].values
    y_data = group['pop'].values

    # Perform linear regression
    params, _ = curve_fit(linear_fit, x_data, y_data)

    # Save the fit parameters to a dictionary
    fits[node] = params

# Save the fits dictionary to a NumPy file
np.save(settings.eula_pop_fits, fits)

