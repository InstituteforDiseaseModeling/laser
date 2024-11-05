import ctypes
import os
import numpy as np
from pkg_resources import resource_filename

lib = None

def init( model ):
    try:
        # Construct the path to the shared library in the same directory
        shared_lib_path = resource_filename('idmlaser_cholera', 'mods/libages.so')

        # Load the shared library
        global lib
        lib = ctypes.CDLL(shared_lib_path)

        lib.update_ages.argtypes = [
            ctypes.c_int64,                  # count
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),      # ages
        ]
        # Define the function signature
        lib.update_ages_contiguous_shards.argtypes = [
            ctypes.c_int64,                  # count
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),      # ages
            ctypes.c_int,                     # delta
            ctypes.c_int                     # tick
        ]

    except Exception as ex:
        print( f"Failed to load libages.so." )

    # Doing report init, nothing to do with ages actually
    model.nodes.add_report_property("S", model.params.ticks, dtype=np.uint32)
    model.nodes.add_report_property("E", model.params.ticks, dtype=np.uint32)
    model.nodes.add_report_property("I", model.params.ticks, dtype=np.uint32)
    model.nodes.add_report_property("W", model.params.ticks, dtype=np.uint32) 
    model.nodes.add_report_property("R", model.params.ticks, dtype=np.uint32) 
    model.nodes.add_report_property("NI", model.params.ticks, dtype=np.uint32) 

delta = 8
def step( model, tick ):
    lib.update_ages_contiguous_shards(
            ctypes.c_int64(model.population.count),
            model.population.age,
            delta,
            tick
        )

