import numpy as np
import numba as nb

global use_nb
use_nb = True
global lib
lib = None

# initialize susceptibility based on age
@nb.njit((nb.uint32, nb.int32[:], nb.uint8[:], nb.uint16[:]), parallel=True)
def initialize_susceptibility(count, dob, susceptibility, susceptibility_timer):

    for i in nb.prange(count):
        susceptibility[i] = 0
        # Initialize everyone's immunity timer uniformly from -5 to +5 years,
        # where negative =0 (to get half population initially susceptible)
        # TBD: Replace this with correct formula
        timer = np.random.randint(-5*365, 5*365)
        if timer <= 0:
            timer = 0
            susceptibility[i] = 1
        else:
            susceptibility_timer[i] = timer

    return

def init( model ):
    initialize_susceptibility( model.population.count, model.population.dob, model.population.susceptibility, model.population.susceptibility_timer )
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
def step(model, tick):

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
