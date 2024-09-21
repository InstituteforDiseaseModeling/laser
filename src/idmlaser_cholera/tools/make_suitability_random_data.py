import numpy as np
import pandas as pd

# Define the dimensions of the CSV
rows = 419  # Number of nodes
columns = 365 * 5  # Time steps (5 years with 365 steps per year)

# Generate random amplitude (between 0.1 and 0.9) for each row
amplitudes = np.random.uniform(0.1, 0.9, size=(rows, 1))

# Generate random phase offsets (between 0 and 2*pi for a full cycle) for each row
phase_offsets = np.random.uniform(0, 2 * np.pi, size=(rows, 1))

# Create a time vector representing the time steps for the sin wave (columns represent days)
time_vector = np.arange(columns)

# Generate sinusoidal data for each row
# Formula: value = amplitude * sin(2 * pi * time / period + phase) + 0.5 (to shift range to 0 to 1)
sin_data = amplitudes * np.sin(2 * np.pi * time_vector / 365 + phase_offsets) + 0.5

# Clip the data to ensure it stays between 0 and 1
sin_data_clipped = np.clip(sin_data, 0, 1)

# Create a DataFrame from the sinusoidal values
df = pd.DataFrame(sin_data_clipped)

# Define the filename for the CSV
csv_filename = "synthetic_psi_data.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_filename, index=False)

csv_filename

