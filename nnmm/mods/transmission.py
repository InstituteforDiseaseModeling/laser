import numpy as np
import numba as nb
import ctypes
import pdb

use_nb = True

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
        lib = ctypes.CDLL('./libtx.so')

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
        use_nb = False
    except Exception as ex:
        print( "Failed to load libtx.so. Will use numba." )
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
    (nb.uint8[:], nb.uint16[:], nb.float32[:], nb.uint8[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:]),
    parallel=True,
    nogil=True,
    cache=True,
)
def tx_inner(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, incidence):
    for i in nb.prange(count):
        susceptibility = susceptibilities[i]
        if susceptibility > 0:
            nodeid = nodeids[i]
            force = susceptibility * forces[nodeid] # force of infection attenuated by personal susceptibility
            if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                susceptibilities[i] = 0.0  # set susceptibility to 0.0
                # set exposure timer for newly infected individuals to a draw from a normal distribution, must be at least 1 day
                etimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(exp_mean, exp_std))))
                incidence[nodeid] += 1

    return


def get_enviro_beta_from_psi( beta_env0, psi ):
    # See https://gilesjohnr.github.io/MOSAIC-docs/model-description.html#eq:system, 4.3.1
    # psi is a numpy array of current suitability values for all nodes
    # Calculate average suitability over time (for simplicity, use a rolling mean or a fixed window)
    window_size = 10  # example window size, adjust as necessary
    psi_avg = np.convolve(psi, np.ones(window_size)/window_size, mode='valid')

    # Calculate environmental transmission rate
    beta_env = beta_env0 * (1 + (psi - psi_avg[-1]) / psi_avg[-1])
    return beta_env 

@nb.njit(
    nb.float64[:](
        nb.uint32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32,
        nb.float32,
        nb.float32,
        nb.float32
    ),
    parallel=True, nogil=True, cache=True
)
def get_enviro_foi(
    new_contagion,
    enviro_contagion, 
    WASH_fraction, 
    psi, 
    enviro_base_decay_rate, 
    zeta, 
    beta_env, 
    kappa
):
    num_nodes = enviro_contagion.shape[0]
    forces_environmental = np.zeros(num_nodes)
    psi_mean = np.mean(psi)
    
    for i in nb.prange(num_nodes):
        # Decay the environmental contagion by the base decay rate
        enviro_contagion[i] *= (1 - enviro_base_decay_rate)
        
        # Add newly shed contagion to the environmental contagion, adjusted by zeta
        enviro_contagion[i] += new_contagion[i] * zeta

        # Apply WASH fraction to reduce environmental contagion
        enviro_contagion[i] *= (1 - WASH_fraction[i])

        # Calculate beta_env_effective using psi
        beta_env_effective = beta_env * (1 + (psi[i] - psi_mean) / psi_mean)
        
        # Calculate the environmental transmission forces
        forces_environmental[i] = beta_env_effective * (enviro_contagion[i] / (kappa + enviro_contagion[i]))
   
    return forces_environmental

def do_transmission_update(model, tick) -> None:

    nodes = model.nodes
    population = model.population

    contagion = nodes.cases[:, tick]    # we will accumulate current infections into this array
    nodeids = population.nodeid[:population.count]  # just look at the active agent indices
    itimers = population.itimer[:population.count] # just look at the active agent indices
    np.add.at(contagion, nodeids[itimers > 0], 1)   # increment by the number of active agents with non-zero itimer

    network = nodes.network
    transfer = (contagion * network).round().astype(np.uint32)
    contagion += transfer.sum(axis=1)   # increment by incoming "migration"
    contagion -= transfer.sum(axis=0)   # decrement by outgoing "migration"

    if True: # contact tx
        # Compute the effective beta considering seasonality
        beta_effective = model.params.beta + model.params.seasonality_factor * np.sin(2 * np.pi * (tick - model.params.seasonality_phase) / 365)

        # Update forces based on contagion and beta_effective
        forces = nodes.forces
        np.multiply(contagion, beta_effective, out=forces)
        np.divide(forces, model.nodes.population[:, tick], out=forces)  # per agent force of infection as a probability

    if True:
        forces_environmental = get_enviro_foi(
            new_contagion=contagion,
            enviro_contagion=model.nodes.enviro_contagion,  # Environmental contagion
            WASH_fraction=model.nodes.WASH_fraction,    # WASH fraction at each node
            psi=model.nodes.psi[:, tick],                # Psi data for each node and timestep
            enviro_base_decay_rate=model.params.enviro_base_decay_rate,  # Decay rate
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

    if use_nb:
        tx_inner(
            population.susceptibility,
            population.nodeid,
            total_forces,
            population.etimer,
            population.count,
            model.params.exp_mean,
            model.params.exp_std,
            model.nodes.incidence[:, tick],
        )
    else:
        lib.tx_inner(
            population.susceptibility,
            population.nodeid,
            total_forces,
            population.etimer,
            population.count,
            model.params.exp_mean,
            model.params.exp_std,
            model.nodes.incidence[:, tick],
            len(total_forces)
        )
    return

