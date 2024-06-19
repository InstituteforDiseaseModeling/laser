import numpy as np
import csv

# Load distance matrix
distances = np.load("engwaldist.npy")

# Load population data
from englwaldata import data as engwal
populations = np.array([engwal.places[name].population[0] for name in engwal.places])

# Gravity algorithm to calculate attraction weights
def gravity_attraction(source_pop, dest_pop, distance, alpha=1.0, beta=1.0):
    return (source_pop * dest_pop) / (distance ** alpha) * beta

# Calculate attraction weights
attraction_weights = gravity_attraction(populations[:, np.newaxis], populations, distances)

# Normalize attraction weights to probabilities
probabilities = attraction_weights / np.sum(attraction_weights, axis=1, keepdims=True)

# Generate cumulative probability arrays
cumulative_probabilities = np.cumsum(probabilities, axis=1)

# Store results in a CSV file
with open("attraction_probabilities2.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for row in cumulative_probabilities:
        writer.writerow(row)

# Function to select destination location based on cumulative probabilities
def select_destination(source_index, random_draw):
    return np.argmax(cumulative_probabilities[source_index] > random_draw)

# Example usage
source_location = 0  # Index of the source location
random_draw = np.random.rand()  # Random number between 0 and 1
destination_location = select_destination(source_location, random_draw)
print("Selected destination location:", destination_location)

