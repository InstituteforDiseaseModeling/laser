import numba as nb
import numpy as np

@nb.njit(
    (nb.uint8[:], nb.uint16[:], nb.float32[:], nb.uint8[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:]),
    parallel=True,
    nogil=True,
    cache=True,
)
def tx_inner_nb(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, incidence):
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


import ctypes

# Load the shared library
lib = ctypes.CDLL('./libtx.so')

# Define the function argument types
lib.tx_inner.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # susceptibilities
    ctypes.POINTER(ctypes.c_uint16), # nodeids
    ctypes.POINTER(ctypes.c_float),  # forces
    ctypes.POINTER(ctypes.c_uint8),  # etimers
    ctypes.c_uint32,                 # count
    ctypes.c_float,                  # exp_mean
    ctypes.c_float,                  # exp_std
    ctypes.POINTER(ctypes.c_uint32)  # incidence
]

def tx_inner_c(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, incidence):
# Call the function
    lib.tx_inner(
        susceptibilities.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        nodeids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        forces.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        etimers.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        count,
        ctypes.c_float(exp_mean),
        ctypes.c_float(exp_std),
        incidence.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    )

    # Check results
    #print("Incidence:", incidence)
    return

def tx_inner(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, incidence):
    #return tx_inner_nb(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, incidence)
    return tx_inner_c(susceptibilities, nodeids, forces, etimers, count, exp_mean, exp_std, incidence)
