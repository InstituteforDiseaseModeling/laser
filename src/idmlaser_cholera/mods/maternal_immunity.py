import numpy as np
import numba as nb
import ctypes
import pdb

# # Maternal Immunity (Waning)
# All newborns come into the world with susceptibility=0. They call get a 6month timer. When that timer hits 0, they become susceptible.

use_nb = True
def init( model, istart, iend ):
    # enable this after adding susceptibility property to the population (see cells below)
    model.population.susceptibility[istart:iend] = 0 # newborns have maternal immunity
    model.population.susceptibility_timer[istart:iend] = int(0.5*365) # 6 months

# Define the function to decrement susceptibility_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True)
def _update_susceptibility_based_on_sus_timer_nb(count, susceptibility_timer, susceptibility):
    for i in nb.prange(count):
        if susceptibility_timer[i] > 0:
            susceptibility_timer[i] -= 1
            if susceptibility_timer[i] == 0:
                susceptibility[i] = 1

try:
# Load the shared library
    lib = ctypes.CDLL('./libmi.so')

# Define the function prototype
    lib.update_susceptibility_based_on_sus_timer.argtypes = [ctypes.c_int32,
                                                             ctypes.POINTER(ctypes.c_uint8),
                                                             ctypes.POINTER(ctypes.c_uint8)]
    lib.update_susceptibility_based_on_sus_timer.restype = None
    use_nb = False
except Exception as ex:
    print( "Failed to load libmi.so. Will use numba." )


"""
def _update_susceptibility_based_on_sus_timer_c(count, susceptibility_timer, susceptibility):
    lib._update_susceptibility_based_on_sus_timer(count,
                                                 susceptibility_timer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                                 susceptibility.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

"""

def do_susceptibility_decay(model, tick):
    if use_nb:
        _update_susceptibility_based_on_sus_timer_nb(model.population.count, model.population.susceptibility_timer, model.population.susceptibility)
    else:
        # These should not be necessary
        susceptibility_timer_ctypes = model.population.susceptibility_timer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) 
        susceptibility_ctypes = model.population.susceptibility.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        lib.update_susceptibility_based_on_sus_timer(
                model.population.count,
                susceptibility_timer_ctypes,
                susceptibility_ctypes
            )
    return
