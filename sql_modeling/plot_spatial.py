import pandas as pd
import matplotlib.pyplot as plt

# Timestep,Node,Susceptible,Infected,Recovered
# Read the CSV file into a DataFrame
df = pd.read_csv('simulation_report.csv')

# Extract unique node values
nodes = df['Node'].unique()

# Plot line graphs for I for each node
for node in nodes:
    node_data = df[df['Node'] == node]
    plt.plot(node_data['Timestep'], node_data['Infected']/(node_data['Infected']+node_data['Susceptible']+node_data['Recovered']), label=f'Node {node}')

# Set labels and title
plt.xlabel('Time')
plt.ylabel('Infected (I)')
plt.title('Line Graphs for Infected (I) by Node')
plt.legend()

# Show the plot
plt.show()

