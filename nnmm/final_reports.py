import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

def save_seird( model ):
    # Assuming the arrays are numpy arrays with shape (3651, 420)
    arrays = {
        'S': model.nodes.S,
        'E': model.nodes.E,
        'I': model.nodes.I,
        'R': model.nodes.R,
        'D': model.nodes.D
    }

    # Save the arrays to an HDF5 file
    with h5py.File('node_arrays.h5', 'w') as hf:
        for name, array in arrays.items():
            hf.create_dataset(name, data=array)

    print("Arrays saved successfully.")

def report( model, initial_populations ):
    metrics = pd.DataFrame(model.metrics, columns=["tick"] + [phase.__name__ for phase in model.phases])
    metrics.head()

    save_seird( model )


# ## Timing Metrics Part II
# 
# Let's take a look at where we spend our processing time.

    plot_columns = metrics.columns[1:]
    sum_columns = metrics[plot_columns].sum()
    print(sum_columns)
    print("=" * 33)
    print(f"Total: {sum_columns.sum():26,}")
    plt.figure(figsize=(8, 8))
    plt.pie(sum_columns, labels=sum_columns.index, autopct="%1.1f%%")
    plt.title("Sum of Each Column")
    plt.show()


# ## Validation - Population Over Time
# 
# Let's make sure that our population is growing over time by plotting the population for a few nodes.

    nodes_to_plot = [0, 1, 2, 3]
    node_population = model.nodes.population[nodes_to_plot, :]

    plt.figure(figsize=(10, 6))
    for i, node in enumerate(nodes_to_plot):
        plt.plot(range(model.params.ticks + 1), node_population[i, :], label=f"Node {node}")

    plt.xlabel("Tick")
    plt.ylabel("Population")
    plt.title("Node Population Over Time")
    plt.legend()
    plt.show()


# ## Validation - Births
# 
# Let's see if our births over time look right. Given a fixed CBR and a growing population, we should generally have more births later in the simulation.

    node_births = model.nodes.births[nodes_to_plot, :]

    plt.figure(figsize=(10, 6))
    for i, node in enumerate(nodes_to_plot):
        plt.plot(range((model.params.ticks + 364) // 365), node_births[i, :], label=f"Node {node}")

    plt.xlabel("Year")
    plt.ylabel("Births")
    plt.title("Node Births Over Time")
    plt.legend()
    plt.show()


# ## Validation - Non-Disease Deaths
# 
# Let's see if our non-disease deaths look right over time.

    node_deaths = model.nodes.deaths[nodes_to_plot, :]

    plt.figure(figsize=(10, 6))
    for i, node in enumerate(nodes_to_plot):
        plt.plot(range((model.params.ticks + 364) // 365), node_deaths[i, :], label=f"Node {node}")

    plt.xlabel("Year")
    plt.ylabel("Deaths")
    plt.title("Node Deaths Over Time")
    plt.legend()
    plt.show()


# ## Cases Over Time

    np.savetxt('cases.csv', model.nodes.cases, delimiter=',', fmt='%d')
    group = 0
    size = 16
    nodes_to_plot = list(range(size*group,size*(group+1)))
    nodes_to_plot = [ 0, 1, 2, 3 ]

    window_start = 0
    window_end = 180

    plt.figure(figsize=(10, 6))
    for i, node in enumerate(nodes_to_plot):
        plt.plot(range(window_start,window_end), model.nodes.cases[i, window_start:window_end], label=f"Node {node}")

    plt.xlabel("Tick")
    plt.ylabel("Cases")
    plt.title("Node Cases Over Time")
    plt.legend()
    plt.show()


# ## Incidence Over Time

    group = 0
    size = 16
    nodes_to_plot = list(range(size*group,size*(group+1)))
    nodes_to_plot = [ 0, 1, 2, 3 ]

    window_start = 0
    window_end = 180

    plt.figure(figsize=(10, 6))
    for i, node in enumerate(nodes_to_plot):
        plt.plot(range(window_start,window_end), model.nodes.incidence[i, window_start:window_end], label=f"Node {node}")

    plt.xlabel("Tick")
    plt.ylabel("Incidence")
    plt.title("Node Incidence Over Time")
    plt.legend()
    plt.show()


    plt.hist(initial_populations)
    plt.xlabel('Population')
    plt.ylabel('Frequency')
    plt.title('Histogram of Initial Populations')
    plt.yscale('log')  # Set y-axis to log scale
    plt.show()

