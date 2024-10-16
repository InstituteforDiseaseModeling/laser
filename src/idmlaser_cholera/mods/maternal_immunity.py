import numpy as np
import numba as nb
import ctypes
import pdb
from pkg_resources import resource_filename

# # Maternal Immunity (Waning)
# All newborns come into the world with susceptibility=0. They call get a 6month timer. When that timer hits 0, they become susceptible.

global use_nb
use_nb = True
global lib
lib = None

def init( model, istart, iend ):
    # enable this after adding susceptibility property to the population (see cells below)
    model.population.susceptibility[istart:iend] = 0 # newborns have maternal immunity
    model.population.susceptibility_timer[istart:iend] = int(0.5*365) # 6 months

try:
    # Load the shared library
    shared_lib_path = resource_filename('idmlaser_cholera', 'mods/libmi.so')
    lib = ctypes.CDLL(shared_lib_path)

    # Define the function prototype
    lib.update_susceptibility_based_on_sus_timer.argtypes = [
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'), # susceptibility_timer
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'), # susceptibility
    ]
        
    lib.update_susceptibility_based_on_sus_timer.restype = None

    lib.update_susceptibility_timer_strided_shards.argtypes = [
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS'), # susceptibility_timer
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'), # susceptibility
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    use_nb = False
    print( "maternal immunity component initialized. Will use compiled C." )
except Exception as ex:
    print( "Failed to load libmi.so. Will use numba." )


# Define the function to decrement susceptibility_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:], nb.uint8, nb.uint8 ), parallel=True)
def _update_susceptibility_based_on_sus_timer_nb(count, susceptibility_timer, susceptibility, tick, delta):
    shard_size = count // delta

    # Determine the start and end indices for the current shard
    shard_index = tick % delta
    start_index = shard_index * shard_size
    end_index = start_index + shard_size

    # Handle the case where the last shard might be slightly larger due to rounding
    if shard_index == delta - 1:
        end_index = count

    # Loop through the current shard
    for i in nb.prange(start_index, end_index):
        if susceptibility_timer[i] > 0:
            susceptibility_timer[i] = max(0, susceptibility_timer[i] - delta)
            if susceptibility_timer[i] <= 0:
                susceptibility[i] = 1

delta = 8
def do_susceptibility_decay(model, tick):

    global lib, use_nb
    if use_nb:
        _update_susceptibility_based_on_sus_timer_nb(model.population.count, model.population.susceptibility_timer, model.population.susceptibility, tick, delta)
    else:
        lib.update_susceptibility_timer_strided_shards(
                model.population.count,
                model.population.susceptibility_timer,
                model.population.susceptibility,
                delta,
                tick
            )
    return
