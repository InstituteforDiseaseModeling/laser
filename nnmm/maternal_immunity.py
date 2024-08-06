import numpy as np
import numba as nb
import ctypes

def add( model, istart, iend ):
    # enable this after adding susceptibility property to the population (see cells below)
    model.population.susceptibility[istart:iend] = 0 # newborns have maternal immunity
    model.population.susceptibility_timer[istart:iend] = int(0.5*365) # 6 months

"""
# Define the function to decrement susceptibility_timer and update susceptibility
@nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:]), parallel=True)
def _update_susceptibility_based_on_sus_timer(count, susceptibility_timer, susceptibility):
    for i in nb.prange(count):
        if susceptibility_timer[i] > 0:
            susceptibility_timer[i] -= 1
            if susceptibility_timer[i] == 0:
                susceptibility[i] = 1
"""


# Load the shared library
lib = ctypes.CDLL('./libmi.so')

# Define the function prototype
lib.update_susceptibility_based_on_sus_timer.argtypes = [ctypes.c_uint32,
                                                         ctypes.POINTER(ctypes.c_uint8),
                                                         ctypes.POINTER(ctypes.c_uint8)]
lib.update_susceptibility_based_on_sus_timer.restype = None

# Example usage

def _update_susceptibility_based_on_sus_timer(count, susceptibility_timer, susceptibility):
    lib.update_susceptibility_based_on_sus_timer(count,
                                                 susceptibility_timer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                                 susceptibility.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))



