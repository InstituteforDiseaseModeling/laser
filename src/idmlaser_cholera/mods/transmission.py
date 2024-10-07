import numpy as np
import numba as nb
import ctypes
import pdb
from pkg_resources import resource_filename

use_nb = True
lib = None
ll_lib = None
psi_means = None
#infected_ids_type = ctypes.POINTER(ctypes.c_uint32)

# Define the maximum number of infections you expect
MAX_INFECTIONS = 100000000  # Adjust this to your expected maximum

# Allocate a flat array for infected IDs
infected_ids_buffer = (ctypes.c_uint32 * (MAX_INFECTIONS))()

# ## Transmission Part I - Setup
# 
# We will add a `network` property to the model to hold the connection weights between the nodes.
# 
# We initialize $n_{ij} = n_{ji} = k \frac {P_i^a \cdot P_j^b} {D_{ij}^c}$
# 
# Then we limit outgoing migration from any one node to `max_frac`.

# In[21]:


# We need to calculate the distances between the centroids of the nodes in northern Nigeria

RE = 6371.0  # Earth radius in km
#delta = np.ones( 419, dtype=np.float32 )*0.5
#delta = 0.5

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

def init( model ):
    initial_populations = model.nodes.population[:,0]
    network = model.nodes.network
    locations = np.zeros((model.nodes.count, 2), dtype=np.float32)

    for i, node in enumerate(model.nodes.nn_nodes.values()):
        (longitude, latitude) = node[1]
        locations[i, 0] = latitude
        locations[i, 1] = longitude
    locations = np.radians(locations)

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
        lib.tx_inner.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # susceptibility
            np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),  # nodeids
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # forces
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # itimers
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # etimers
            ctypes.c_uint32,                                                        # count
            ctypes.c_float,                                                           # exp_mean
            ctypes.c_float,                                                           # exp_std
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),  # incidence,
            ctypes.c_uint32,                                                        # num_nodes
        ]
        lib.tx_inner_nodes.argtypes = [
            ctypes.c_uint32,                                                        # count
            ctypes.c_uint32,                                                        # num_nodes
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # susceptibility
            np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),   # etimers
            np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'),  # new_infections,
            ctypes.c_float,                                                           # exp_mean
            ctypes.POINTER(ctypes.c_uint32)  # new_ids_out (pointer to uint32)
        ]
        lib.report.argtypes = [
            ctypes.c_int64,                  # count
            ctypes.c_int,                     # num_nodes
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),      # age
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
def tx_inner(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, new_infections_by_node ):
    for i in nb.prange(count):
        susceptibility = susceptibilities[i]
        if susceptibility > 0:
            nodeid = nodeids[i]
            if new_infections_by_node[nodeid] > 0:
                #etimers[i] = 2 # didn't make anything faster
                etimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(exp_mean, exp_std))))
                susceptibilities[i] = 0.0  # set susceptibility to 0.0
                # This is probably blocking; we need to send each node to its own process
                new_infections_by_node[nodeid] -= 1

    return # incidence


def get_enviro_beta_from_psi( beta_env0, psi ):
    # See https://gilesjohnr.github.io/MOSAIC-docs/model-description.html#eq:system, 4.3.1
    # psi is a numpy array of current suitability values for all nodes
    # Calculate average suitability over time (for simplicity, use a rolling mean or a fixed window)
    window_size = 10  # example window size, adjust as necessary
    psi_avg = np.convolve(psi, np.ones(window_size)/window_size, mode='valid')

    # Calculate environmental transmission rate
    beta_env = beta_env0 * (1 + (psi - psi_avg[-1]) / psi_avg[-1])
    return beta_env 

# Sometimes I think it might be faster not numba-ing this function but I
# want to try a compiled C version of it at some point.
@nb.njit(
    nb.float32[:](
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32,
        nb.float32,
        nb.float32
    ),
    parallel=True
    #, nogil=True, cache=True
)
def get_enviro_foi(
    new_contagion,
    enviro_contagion, 
    WASH_fraction, 
    psi, 
    psi_mean,
    enviro_base_decay_rate, 
    zeta, 
    beta_env, 
    kappa
):
    """
    Calculate the environmental force of infection (FOI) for each node in the model, considering various
    factors such as environmental decay, WASH practices, environmental suitability (psi), and transmission
    parameters.

    Parameters
    ----------
    new_contagion : np.ndarray, shape (num_nodes,), dtype=float32
        Array representing the amount of newly shed contagion for each node at the current timestep.

    enviro_contagion : np.ndarray, shape (num_nodes,), dtype=float32
        Array representing the existing environmental contagion for each node. This value will be updated
        based on decay rates, new contagion, WASH fraction, and other factors.

    WASH_fraction : np.ndarray, shape (num_nodes,), dtype=float32
        Array representing the effect of water, sanitation, and hygiene (WASH) interventions at each node.
        Values should be between 0 and 1, where 0 means no reduction in contagion and 1 means complete
        elimination of contagion.

    psi : np.ndarray, shape (num_nodes,), dtype=float32
        Array representing the environmental suitability for pathogen persistence at each node. Values
        typically range from 0 to 1, where higher values indicate more favorable conditions for the pathogen.

    enviro_base_decay_rate : float32
        The base decay rate of environmental contagion, representing the natural reduction of contagion over
        time. This should be a value between 0 and 1, where 0 means no decay and 1 means complete decay.

    zeta : float32
        A scaling factor applied to the newly shed contagion when adding it to the environmental contagion.
        This allows for adjustment of how much new contagion contributes to the overall environmental
        contagion.

    beta_env : float32
        A transmission parameter that represents the baseline rate at which environmental contagion
        contributes to the force of infection. This value is adjusted by the psi suitability index.

    kappa : float32
        A threshold parameter used to control the non-linear effect of environmental contagion on the force
        of infection. Larger values of kappa reduce the influence of environmental contagion on transmission
        forces.

    Returns
    -------
    forces_environmental : np.ndarray, shape (num_nodes,), dtype=float32
        Array representing the calculated environmental force of infection for each node. This is based on
        the updated environmental contagion and transmission parameters.

    Notes
    -----
    The function performs the following steps:
    1. Decay the existing environmental contagion using the base decay rate.
    2. Add newly shed contagion to the environmental contagion, scaled by the zeta factor.
    3. Apply WASH fraction to reduce the environmental contagion at each node.
    4. Adjust the transmission rate (`beta_env`) using the psi suitability index for each node.
    5. Calculate the force of infection for each node as a function of the environmental contagion and the
       kappa parameter, which introduces a saturation effect at high levels of contagion.

    The resulting `forces_environmental` array provides the force of infection for each node, which can be
    used to model transmission dynamics in an agent-based or compartmental model.
    """

    invalid_psi = psi[(psi < 0) | (psi > 1.0)]  # Find any invalid values in the array
    if invalid_psi.size > 0:
        raise ValueError(f"psi contains invalid values: {invalid_psi}. Each value must be between 0 and 1.0")

    num_nodes = enviro_contagion.shape[0]
    forces_environmental = np.zeros(num_nodes, dtype=np.float32)

    #decay_rate_mult = (1 - enviro_base_decay_rate)
    for node in nb.prange(num_nodes):
        # Decay the environmental contagion by the base decay rate
        enviro_contagion[node] *= (1 - enviro_base_decay_rate[node])
        #enviro_contagion[node] *= decay_rate_mult 
        
        # Add newly shed contagion to the environmental contagion, adjusted by zeta
        enviro_contagion[node] += new_contagion[node] * zeta

        # Apply WASH fraction to reduce environmental contagion
        enviro_contagion[node] *= (1 - WASH_fraction[node])

        # Calculate beta_env_effective using psi
        beta_env_effective = beta_env * (1 + (psi[node] - psi_mean[node]) / psi_mean[node])

        # Calculate the environmental transmission forces
        forces_environmental[node] = beta_env_effective * (enviro_contagion[node] / (kappa + enviro_contagion[node]))
   
    return forces_environmental

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
    new_infections = np.zeros(num_nodes, dtype=np.uint16)

    # Cap the total forces at 1.0 using np.minimum
    capped_forces = np.minimum(total_forces, 1.0)

    # Calculate new infections in a vectorized way
    try:
        new_infections = np.random.binomial(susceptibles, capped_forces).astype(np.uint16)
    except Exception as ex:
        pdb.set_trace()
    #print( f"new_infections = {new_infections}" )


    return new_infections

def do_transmission_update(model, tick) -> None:

    delta = 7
    nodes = model.nodes
    population = model.population

    global lib
    lib.report(
        len(population),
        len(nodes),
        model.population.age,
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

    contagion = nodes.cases[:, tick].astype(np.float32)    # we will accumulate current infections into this array
    contagion += model.nodes.I[tick]

    network = nodes.network
    transfer = (contagion * network).round().astype(np.uint32)
    contagion += transfer.sum(axis=1)   # increment by incoming "migration"
    contagion -= transfer.sum(axis=0)   # decrement by outgoing "migration"
    contagion *= delta

    global psi_means
    if psi_means is None:
        psi_means = np.mean(model.nodes.psi, axis=1)

    # Code-based ways of toggling contact and enviro transmission routes on and off during perf investigations.
    if True: # contact tx
        # Compute the effective beta considering seasonality
        beta_effective = model.params.beta + model.params.seasonality_factor * np.sin(2 * np.pi * (tick - model.params.seasonality_phase) / 365)
        #beta_effective = model.params.beta

        # Update forces based on contagion and beta_effective
        forces = nodes.forces
        np.multiply(contagion, beta_effective, out=forces)
        np.divide(forces, model.nodes.population[:, tick], out=forces)  # per agent force of infection as a probability

    delta = model.params.delta_min + model.nodes.psi[:,tick] * (model.params.delta_max - model.params.delta_min)

    if True:
        forces_environmental = get_enviro_foi(
            new_contagion=contagion,
            enviro_contagion=model.nodes.enviro_contagion,  # Environmental contagion
            WASH_fraction=model.nodes.WASH_fraction,    # WASH fraction at each node
            psi=model.nodes.psi[:, tick],                # Psi data for each node and timestep
            psi_mean=psi_means,
            enviro_base_decay_rate=delta, # model.params.enviro_base_decay_rate,  # Decay rate
            zeta=model.params.zeta,             # Shedding multiplier
            beta_env=model.params.beta_env,     # Base environmental transmission rate
            kappa=model.params.kappa            # Environmental scaling factor
        )

    # Combine the contact transmission forces with the environmental transmission forces
    # `forces` are the contact transmission forces calculated elsewhere
    # `forces_environmental` are the environmental transmission forces computed in this section
    total_forces = (forces + forces_environmental).astype(np.float32)
    #total_forces = (forces_environmental).astype(np.float32) # enviro only
    #total_forces = forces

    new_infections = calculate_new_infections_by_node(total_forces, model.nodes.S[tick])
    

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

"""
Latest design thought:

    - Skip current census reporting in ages code.
    - Do census to get number of active infections by node, and number of susceptibles by node. Probably in C. save this somewhere.
    - Calculate foi in Python
    - Calculate number of new infections by node. Python.
    - Do new infections.
    - See if this can all be time-sharded. Divide population into 7 or 8. Process 1/7 of population each day of the week.
"""
