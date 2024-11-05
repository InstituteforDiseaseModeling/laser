import numpy as np
import numba as nb

# ## Initial Infections
# 
# We choose up to 10 infections per node initially. Another option would be to choose the initial number of infections based on estimated prevalence.
# 
# **Note:** this function has a known bug based on using Numba to optimize the function. [Numba only exposes an atomic decrement in CUDA implementations](https://numba.readthedocs.io/en/stable/cuda/intrinsics.html) so on the CPU it is possible that more than one thread decides there are still agents to initialize as infected and the total count of initial infections is more than specified.
# 
# **_TODO:_** After checking to see if there are still infections to be initialized in this node, consider checking the susceptibility of the agent as well, i.e., only infect susceptible agents.
# 
# **_TODO_:** Fix this with a compiled C/C++ function and [OpenMP 'atomic update'](https://www.openmp.org/spec-html/5.0/openmpsu95.html).

# In[20]:


# initial_infections = np.random.randint(0, 11, model.nodes.count, dtype=np.uint32)

@nb.njit((nb.uint32, nb.uint32[:], nb.uint16[:], nb.uint8[:], nb.uint8[:], nb.float32, nb.float32), parallel=True)
def initialize_infections(count, infections, nodeid, itimer, sus, inf_mean, inf_std):

    for i in nb.prange(count):
        if infections[nodeid[i]] > 0 and sus[i] == 1:
            infections[nodeid[i]] -= 1
            itimer[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std)))) # must be at least 1 day

    return

def init( model, zero_nodes=None ):
    #print( zero_nodes )
    if zero_nodes is not None and len(zero_nodes)>0:
        # Step 1: Initialize the output array as a copy of seed_values
        output_array = np.zeros(len(model.nodes.initial_infections), dtype=np.uint32)

        # Step 2: Use the eliminated_nodes array to mask the output
        output_array[zero_nodes] = model.nodes.initial_infections[zero_nodes]
        print( f"Re-seeding with this init_prev array: {output_array}" )
    else:
        import copy
        output_array = copy.deepcopy( model.nodes.initial_infections )

    return initialize_infections(
            np.uint32(model.population.count),
            output_array, # model.nodes.initial_infections,
            model.population.nodeid,
            model.population.itimer,
            model.population.susceptibility,
            model.params.inf_mean,
            model.params.inf_std
        )
