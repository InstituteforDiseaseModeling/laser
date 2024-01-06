import csv
import matplotlib.pyplot as plt

# Read the CSV file
with open('simulation_report.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    data = list(reader)

# Extract data for plotting
timesteps = [int(row[0]) for row in data]
susceptible = [int(row[1]) for row in data]
infected = [int(row[2]) for row in data]
recovered = [int(row[3]) for row in data]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(timesteps, susceptible, label='Susceptible', linestyle='--', marker='o')
plt.plot(timesteps, infected, label='Infected', linestyle='-', marker='o')
plt.plot(timesteps, recovered, label='Recovered', linestyle='-.', marker='o')

plt.xlabel('Timestep')
plt.ylabel('Number of Agents')
plt.title('SIRS Model Simulation')
plt.legend()
plt.grid(True)
plt.show()

