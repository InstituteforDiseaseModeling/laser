import pandas as pd
import matplotlib.pyplot as plt

def plot_sir_curves(csv_file, node_id=0):
    # Load data from CSV file
    df = pd.read_csv(csv_file)

    # Filter data for the specified node
    node_data = df[df['Node'] == node_id]

    # Plot SIR curves
    plt.plot(node_data['Timestep'], node_data['Susceptible'], label='Susceptible')
    plt.plot(node_data['Timestep'], node_data['Infected'], label='Infected')
    plt.plot(node_data['Timestep'], node_data['Recovered'], label='Recovered')

    # Set plot labels and title
    plt.xlabel('Timestamp')
    plt.ylabel('Population')
    plt.title(f'SIR Curves for Node {node_id}')
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    import sys

    # Get CSV file and node_id from command line arguments
    csv_file = 'simulation_output.csv'  # Replace with the actual file path
    node_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Plot SIR curves for the specified node
    plot_sir_curves(csv_file, node_id)

