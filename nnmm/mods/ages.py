import ctypes
import os
import numpy as np

# Get the directory where the current script is located
#script_dir = os.path.dirname(os.path.abspath(__file__))
from pkg_resources import resource_filename

# Construct the path to the shared library in the same directory
shared_lib_path = os.path.join(".", 'libages.so')
#shared_lib_path = resource_filename('idmlaser', 'update_ages.so')

# Load the shared library
lib = ctypes.CDLL(shared_lib_path)

# Define the function signature
lib.update_ages.argtypes = [
    ctypes.c_uint32,                  # count
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
]

def update_ages( model, tick ):
    lib.update_ages(
            model.population.count,
            model.population.age
        )

