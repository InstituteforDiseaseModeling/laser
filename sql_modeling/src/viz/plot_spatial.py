import pandas as pd
import matplotlib.pyplot as plt

# Timestep,Node,Susceptible,Infected,Recovered
# Read the CSV file into a DataFrame
df = pd.read_csv('simulation_output.csv')

# Extract unique node values
nodes = df['Node'].unique()

# Plot line graphs for I for each node
prevalences = []
for node in nodes:
    node_data = df[df['Node'] == node]
#    plt.plot(node_data['Timestep'], , label=f'Node {node}')
    prevalences.append( node_data['Infected']/(node_data['Infected']+node_data['Susceptible']+node_data['Recovered']) )

def plot_together():
    #plt.stackplot(node_data['Timestep'], *prevalences)
    for i, line in enumerate(prevalences):
        plt.plot(node_data['Timestep'], prevalences[i], label=f"Node {i}" )

    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Infected (I)')


def plot_grid():
    # Create a single-column stacked grid of subplots
    fig, axs = plt.subplots(len(prevalences), 1, sharex=True, figsize=(6, 3 * len(prevalences)))

    # Plot each line in a separate subplot
    for i, line in enumerate(prevalences):
        axs[i].plot(node_data['Timestep'], line, label=f'Line {i+1}')
        axs[i].set_title(f'Subplot {i+1}')
        axs[i].legend()

    # Add labels to the last subplot
    axs[-1].set_xlabel('Time')
    axs[-1].set_ylabel('Infected')

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

prevalences = prevalences[-25:]
plot_grid()
#plot_together()

plt.title('Line Graphs for Infected (I) by Node')
plt.legend()

# Show the plot
plt.show()

