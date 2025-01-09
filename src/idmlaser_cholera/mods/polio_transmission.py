import numpy as np
import numba as nb
import ctypes
import pdb
from pkg_resources import resource_filename
from idmlaser_cholera.utils import viz_2D, load_csv_maybe_header

use_nb = True
lib = None
ll_lib = None
seasonal_contact_data = None

# Define the maximum number of infections you expect
MAX_INFECTIONS = 100000000  # Adjust this to your expected maximum

# Allocate a flat array for infected IDs
infected_ids_buffer = (ctypes.c_uint32 * (MAX_INFECTIONS))()

# We need to calculate the distances between the centroids of the nodes in northern Nigeria

RE = 6371.0  # Earth radius in km

# Some of these are inputs and some are outputs
# static inputs
def get_additive_seasonality_effect( model, tick ):
    # this line is a potential backup if no data s provided, but only for "I'm a new user and want this thing to just run"
    if seasonal_contact_data is not None:
        return 0.2*seasonal_contact_data[tick//7 % 52]
    else:
        return model.params.seasonality_factor * np.sin(2 * np.pi * (tick - model.params.seasonality_phase) / 365)


def calc_distance(lat1, lon1, lat2, lon2):
    # convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # haversine formula (https://en.wikipedia.org/wiki/Haversine_formula)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    d = RE * c
    return d

def init( model, manifest ):
    model.nodes.add_vector_property("network", model.nodes.count, dtype=np.float32)

    # report outputs
    model.nodes.add_vector_property("cases", model.params.ticks, dtype=np.uint32)
    model.nodes.add_vector_property("incidence", model.params.ticks, dtype=np.uint32)

    # transient for calculations
    model.nodes.add_scalar_property("forces", dtype=np.float32)
    model.nodes.add_scalar_property("enviro_contagion", dtype=np.float32)

    initial_populations = model.nodes.population[0]
    if initial_populations.sum() == 0:
        raise ValueError( f"Initial Population empty in transmission init." )
    network = model.nodes.network
    locations = np.zeros((model.nodes.count, 2), dtype=np.float32)

    for i, node in enumerate(model.nodes.nn_nodes.values()):
        (longitude, latitude) = node[1]
        locations[i, 0] = latitude
        locations[i, 1] = longitude
    #locations = np.radians(locations)

    # Initialize tx_hetero_factor to values drawn from 0.5 to 2.0 (for now)
    # untested
    model.population.tx_hetero_factor = np.random.uniform(0.5, 2.0, size=model.population.capacity)

# TODO: Consider keeping the distances and periodically recalculating the network values as the populations change
    a = model.params.a
    b = model.params.b
    c = model.params.c
    k = model.params.k
    from tqdm import tqdm
    for i in tqdm(range(model.nodes.count)):
        popi = initial_populations[i]
        for j in range(i+1, model.nodes.count):
            popj = initial_populations[j]
            network[i,j] = network[j,i] = k * (popi**a) * (popj**b) / (calc_distance(*locations[i], *locations[j])**c)
    network /= np.power(initial_populations.sum(), c)    # normalize by total population^2

    #network *= 1000

    print(f"Upper left corner of network looks like this (before limiting to max_frac):\n{network[:4,:4]}")

    max_frac = model.params.max_frac
    for row in range(network.shape[0]):
        if (maximum := network[row].sum()) > max_frac:
            network[row] *= max_frac / maximum

    print(f"Upper left corner of network looks like this (after limiting to max_frac):\n{network[:4,:4]}")

    try:
        shared_lib_path = resource_filename('idmlaser_cholera', 'mods/libtx.so')
        global lib
        lib = ctypes.CDLL(shared_lib_path)

        # Define the argument types for the C function
        lib.tx_inner_nodes.argtypes = [
            ctypes.c_uint32,                                                        # count
            ctypes.c_uint32,                                                        # num_nodes
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # susceptibility
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # etimers
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # new_infections,
            ctypes.c_float,                                                           # exp_mean
            ctypes.POINTER(ctypes.c_uint32)  # new_ids_out (pointer to uint32)
        ]
        lib.report.argtypes = [
            ctypes.c_int64,                  # count
            ctypes.c_int,                     # num_nodes
            np.ctypeslib.ndpointer(dtype=np.uint16, flags='C_CONTIGUOUS'),      # node
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),      # infectious_timer
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),      # incubation_timer
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),      # immunity
            np.ctypeslib.ndpointer(dtype=np.uint16, flags='C_CONTIGUOUS'),      # susceptibility_timer
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),    # expected_lifespan
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # infectious_count
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # incubating_count
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # susceptible_count
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # waning_count 
            np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # recovered_count 
            ctypes.c_int,                     # delta
            ctypes.c_int                     # delta
        ]
        global use_nb
        use_nb = False
    except Exception as ex:
        print( f"Failed to load {shared_lib_path}. Will use numba." )

    try:
        shared_lib_path = resource_filename('idmlaser_cholera', 'mods/libll_reporter.so')
        global ll_lib
        ll_lib = ctypes.CDLL(shared_lib_path)

        # Define the argument types for the C functions
        filename = 'incidence_linelist.bin' # put in manifest
        ll_lib.init_writer.argtypes = [ctypes.c_char_p]
        ll_lib.write_record.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        ll_lib.write_records_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # agent_ids array
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # ages_at_infection array
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # times_at_infection array
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # nodes_at_infection array
            ctypes.c_size_t  # num_records
        ]
        ll_lib.close_writer.argtypes = []
        ll_lib.init_writer(filename.encode('utf-8'))  
    except Exception as ex:
        print( f"Failed to load {shared_lib_path}. No backup." )

    try:
        global seasonal_contact_data 
        seasonal_contact_data = load_csv_maybe_header( manifest.seasonal_dynamics )
        if seasonal_contact_data.shape[0] > model.nodes.count:
            seasonal_contact_data = seasonal_contact_data[:model.nodes.count]
    except Exception as ex:
        print( str( ex ) )
        print( f"WARNING: ***{manifest.seasonal_dynamics} either not found or not parsed correctly. Proceeding with synthetic sinusoidal seasonality***." )
    if model.params.viz:
        viz_2D( model, seasonal_contact_data, "Seasonal Contact Factor", "timestep", "node" )
    seasonal_contact_data = seasonal_contact_data.T
    
    return


# ## Transmission Part II - Tick/Step Processing Phase
# 
# On a tick we accumulate the contagion in each node - currently 1 unit per infectious agent - with `np.add.at()` ([documentation](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)).
# 
# We calculate the incoming and outgoing contagion by multiplying by the network connection values.
# 
# We determine the force of infection per agent in each node by multiplying by the seasonally adjusted $\beta$ and normalizing by the node population. $\beta_{eff}$ is currently a scalar but it would be trivial to add a `beta` property to `model.nodes` and have per node betas.
# 
# We then visit each susceptible agent and draw against the force of infection to determine transmission and draw for duration of infection if transmission occurs.
# 
# We will also track incidence by node and tick.

# In[22]:


@nb.njit(
    (
        nb.uint8[:], # susceptibilities, 
        nb.uint16[:], # nodeids, 
        nb.float32[:], # forces, 
        nb.uint8[:], # etimers, 
        nb.int64, # nb.uint32, # count, 
        nb.float32, # exp_mean, 
        nb.float32, # exp_std, 
        nb.uint16[:], # expected_incidence, 
    ),
    parallel=True,
    nogil=True,
    cache=True,
)
def tx_inner_nodes(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, new_infections_by_node ):
    num_nodes = len(new_infections_by_node)  # Assume number of nodes is the length of new_infections_by_node

    for nodeid in nb.prange(num_nodes):  # Parallelize by node
        infections_left = new_infections_by_node[nodeid]  # New infections required for this node

        if infections_left > 0:
            for i in range(count):  # Loop over all agents
                if infections_left == 0:
                    break  # Stop once we've infected the required number of agents

                if nodeids[i] == nodeid and susceptibilities[i] > 0:  # Check if the agent belongs to the current node and is susceptible
                    # Infect the agent
                    etimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(exp_mean, exp_std))))
                    susceptibilities[i] = 0.0  # Set susceptibility to 0
                    infections_left -= 1  # Decrement the infections count for this node

            # Update the number of remaining infections for this node in case the node wasn't fully exhausted
            new_infections_by_node[nodeid] = infections_left

    return

def calculate_new_infections_by_node(total_forces, susceptibles):
    """
    Calculate new infections per node.

    Parameters:
    - total_forces: array of FOI (force of infection) for each node.
    - model.nodes.S: 2D array where each row corresponds to a node and contains the number of susceptibles in that node.

    Returns:
    - new_infections: array of new infections per node.
    """

    # Get the number of nodes (assuming each row of model.nodes.S corresponds to a node)
    num_nodes = len(susceptibles)

    # Initialize an array to hold the new infections for each node
    new_infections = np.zeros(num_nodes, dtype=np.uint32)

    # Cap the total forces at 1.0 using np.minimum
    capped_forces = np.minimum(total_forces, 1.0)
    capped_forces = np.maximum(capped_forces, 0.0)
    capped_forces = np.array(capped_forces, dtype=np.float64)
    susceptibles = np.array(susceptibles, dtype=np.uint32)

    # Calculate new infections in a vectorized way
    try:
        new_infections = np.random.binomial(susceptibles, capped_forces).astype(np.uint32)
    except Exception as ex:
        print( str( ex ) )
        pdb.set_trace()
    #print( f"new_infections = {new_infections}" )

    return new_infections

def step(model, tick) -> None:

    delta = 1
    nodes = model.nodes
    population = model.population

    global lib
    lib.report(
        len(population),
        len(nodes),
        #model.population.age,
        model.population.nodeid,
        model.population.itimer,
        model.population.etimer,
        model.population.susceptibility,
        model.population.susceptibility_timer,
        model.population.dod,
        model.nodes.S[tick],
        model.nodes.E[tick],
        model.nodes.I[tick],
        model.nodes.W[tick],
        model.nodes.R[tick],
        delta,
        tick
    )

    contagion = nodes.cases[tick].astype(np.float32)    # we will accumulate current infections into this array
    """
    print( f"RAW {model.nodes.S[tick]=}" )
    print( f"RAW {model.nodes.E[tick]=}" )
    print( f"RAW {model.nodes.I[tick]=}" )
    print( f"RAW {model.nodes.W[tick]=}" )
    print( f"RAW {model.nodes.R[tick]=}" )
    """
    contagion += model.nodes.I[tick]
    #print( f"RAW {contagion=}" )

    network = nodes.network
    transfer = (contagion * network).round().astype(np.uint32)
    # The migration functions seem to be able to make the contagion negative in certain contexts
    contagion += transfer.sum(axis=1)   # increment by incoming "migration"
    contagion -= transfer.sum(axis=0)   # decrement by outgoing "migration"
    
    # Compute the effective beta considering seasonality
    beta_effective = model.params.beta + get_additive_seasonality_effect( model, tick )
    #if np.any( beta_effective < 0 ):
        #raise ValueError( "beta went negative after subtracting seasonality." )
    #beta_effective = model.params.beta

    #print( f"{contagion=}" )
    # Update forces based on contagion and beta_effective
    forces = nodes.forces
    np.multiply(contagion, beta_effective, out=forces)
    #print( f"{forces=}" )
    np.divide(forces, model.nodes.population[tick], out=forces)  # per agent force of infection as a probability
    #print( f"normalized {forces=}" )

    # Combine the contact transmission forces with the environmental transmission forces
    # `forces` are the contact transmission forces calculated elsewhere
    # `forces_environmental` are the environmental transmission forces computed in this section
    total_forces = forces

    new_infections = calculate_new_infections_by_node(total_forces, model.nodes.S[tick])
    model.nodes.NI[tick] = new_infections 
    #print( f"{new_infections=}" )
    

    total_infections = np.sum(new_infections)
    #print( f"total new infections={total_infections}" )
    if total_infections > MAX_INFECTIONS:
        raise ValueError( f"Number of new infections ({total_infections}) > than allocated array size ({MAX_INFECTIONS})!" )

    if use_nb:
        tx_inner_nodes(
            population.susceptibility,
            population.nodeid,
            total_forces,
            population.etimer,
            population.count,
            model.params.exp_mean,
            model.params.exp_std,
            new_infections 
            #model.nodes.incidence[:, tick],
        )
    else:
        num_nodes = len(new_infections)  # Assume number of nodes is the length of new_infections_by_node
        lib.tx_inner_nodes(
            population.count,
            num_nodes,
            population.susceptibility,# uint8_t *susceptibility,
            population.etimer,# unsigned char  * incubation_timer,
            new_infections, # int * new_infections_array,
            model.params.exp_mean, # unsigned char incubation_period_constant
            infected_ids_buffer
        )
        # Call our ctypes module function to report these ids, and the current time, and agent ages and nodes to the linelist reporter

        def report_linelist():
            current_index = 0
            global ll_lib
            infected_ids_arr = np.ctypeslib.as_array(infected_ids_buffer)[:total_infections]
            ages_at_infection = model.population.age[infected_ids_arr].astype( np.uint32 )
            ll_lib.write_records_batch(
                infected_ids_arr,
                ages_at_infection,
                np.ones( total_infections ).astype( np.uint32 ) * tick,
                np.repeat(np.arange(num_nodes), new_infections).astype( np.uint32 ),
                total_infections
            )
        #report_linelist()

    return

