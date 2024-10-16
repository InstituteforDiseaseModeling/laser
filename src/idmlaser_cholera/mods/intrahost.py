import numba as nb
import numpy as np
import ctypes
from pkg_resources import resource_filename

use_nb = True
lib = None
delta = 2

def init( model ):
    global use_nb, lib
    try:
    # Load the shared library
        shared_lib_path = resource_filename('idmlaser_cholera', 'mods/libintrahost.so')
        lib = ctypes.CDLL(shared_lib_path)

        # Define the argument types for the C function
        lib.exposure_update.argtypes = [
            ctypes.c_uint32,               # count
            np.ctypeslib.ndpointer(dtype=np.uint8),    # etimer
            np.ctypeslib.ndpointer(dtype=np.uint8),    # itimer
            ctypes.c_float,                # inf_mean
            ctypes.c_float,                # inf_std
            ctypes.c_uint32,               # tick
            ctypes.c_uint32                # delta
        ]
        
        lib.infection_update.argtypes = [
            ctypes.c_uint32,               # count
            np.ctypeslib.ndpointer(dtype=np.uint8),    # itimer
            np.ctypeslib.ndpointer(dtype=np.uint16),   # sus_timer
            np.ctypeslib.ndpointer(dtype=np.uint8),    # sus
            ctypes.c_uint32,               # tick
            ctypes.c_uint32                # delta
        ]

        lib.infection_update_contiguous.argtypes = [
            ctypes.c_uint32,               # count
            np.ctypeslib.ndpointer(dtype=np.uint8),    # itimer
            np.ctypeslib.ndpointer(dtype=np.uint16),   # sus_timer
            np.ctypeslib.ndpointer(dtype=np.uint8),    # sus
            ctypes.c_uint32,               # tick
            ctypes.c_uint32                # delta
        ]
        use_nb = False
    except Exception as ex:
        print( "Couldn't load exposure_update.so. Using numba." )


# ## Incubation and Infection
# 
# We will add incubation timer (`etimer` for "exposure timer") and an infection (or infectious) timer (`itimer`) properties to the model population. A `uint8` counting down from as much as 255 days will be more than enough.
# 
# We wrap a Numba compiled function using all available cores in the infection update function, extracting the `count` and `itimer` values the JITted function needs.
# 
# Similarly, we wrap a Numba compiled function using all available cores in the exposure update function, extracting the values the JITted function needs from the `model` object.

# In[13]:


@nb.njit((
    nb.uint32,
    nb.uint8[:],
    nb.uint8[:],
    nb.uint16[:],
    nb.uint8[:],
    nb.float32, 
    nb.float32,
    nb.uint32),
    parallel=True)
def _exposure_update(count, etimer, itimer, sus_timer, sus, inf_mean, inf_std, tick):
    #start_index = tick % 2

    #for i in nb.prange(start_index, count, 2):
    for i in nb.prange(count):
        if etimer[i] > 0:
            etimer[i] -= 1
            if etimer[i] == 0:
                itimer[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std))))    # must be at least 1 day
    return

def step1(model, tick):

    #global lib
    #if use_nb:
    _exposure_update(nb.uint32(model.population.count), model.population.etimer, model.population.itimer, model.population.susceptibility_timer, model.population.susceptibility, model.params.inf_mean, model.params.inf_std, tick)
    #else:
    #    lib.exposure_update(model.population.count, model.population.etimer, model.population.itimer, model.params.inf_mean, model.params.inf_std, tick, delta)

    return


@nb.njit((nb.uint32, nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.uint32 ), parallel=True)
def _infection_update(count, itimer, sus_timer, susceptibility, tick):
    """
    Updates infection and susceptibility timers for a population of agents.

    Parameters:
    -----------
    count : uint32
        The number of agents to process.
    itimer : uint8[:]
        An array of infection timers, where each element represents the remaining time (in days)
        until the infection ends for a given agent. A value of 0 indicates that the agent is no longer infected.
    sus_timer : uint16[:]
        An array of susceptibility timers, where each element represents the time (in days) until
        the agent becomes susceptible again after recovering from an infection.
    susceptibility : uint8[:]
        An array representing the susceptibility status of each agent. A value of 1 indicates that
        the agent is susceptible to infection, while 0 indicates that the agent is not susceptible.

    Behavior:
    ---------
    - For each infected agent (i.e., those with `itimer[i] > 0`), the infection timer (`itimer[i]`) is decremented by 1 day.
    - If the infection timer reaches 0, the agent recovers, their susceptibility is set to 0, and their susceptibility timer (`sus_timer[i]`) is set to a random value between 3 and 6 years (1095 to 2190 days).
    - TBD: Replace the uniform draw from 3 to 6 years with appropriate calculation.

    Notes:
    ------
    - This function is optimized using Numba's Just-In-Time (JIT) compilation for parallel execution, which improves performance when processing large populations.
    """
    for i in nb.prange(count):
        if itimer[i] > 0:
            #itimer[i] -= 1
            itimer[i] = max( itimer[i]-delta, 0 )
            if itimer[i] <= 0:
                susceptibility[i] = 0
                sus_timer[i] = np.random.randint(2*365, 4*365) # waning

    return

def step2(model, tick):

    global lib
    if True: # use_nb:
        if tick % delta == 0:
            _infection_update(nb.uint32(model.population.count), model.population.itimer, model.population.susceptibility_timer, model.population.susceptibility, tick)
    else:
        lib.infection_update_contiguous(model.population.count, model.population.itimer, model.population.susceptibility_timer, model.population.susceptibility, tick, delta)

    return

def step(model, tick):
    #do_exposure_update( model, tick )
    #do_infection_update( model, tick )
    _exposure_update(nb.uint32(model.population.count), model.population.etimer, model.population.itimer, model.population.susceptibility_timer, model.population.susceptibility, model.params.inf_mean, model.params.inf_std, tick)
    if tick % delta == 0:
        _infection_update(nb.uint32(model.population.count), model.population.itimer, model.population.susceptibility_timer, model.population.susceptibility, tick)

