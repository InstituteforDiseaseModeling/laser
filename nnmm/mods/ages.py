import ctypes
import os
import numpy as np
import pdb

# Get the directory where the current script is located
#script_dir = os.path.dirname(os.path.abspath(__file__))
from pkg_resources import resource_filename

try:
    # Construct the path to the shared library in the same directory
    shared_lib_path = os.path.join(".", 'libages.so')
    #shared_lib_path = resource_filename('idmlaser', 'update_ages.so')

    # Load the shared library
    lib = ctypes.CDLL(shared_lib_path)

    lib.update_ages.argtypes = [
        ctypes.c_int64,                  # count
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),      # ages
    ]
    # Define the function signature
    lib.update_ages_and_report.argtypes = [
        ctypes.c_int64,                  # count
        ctypes.c_int,                     # num_nodes
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),      # ages
        np.ctypeslib.ndpointer(dtype=np.uint16, flags='C_CONTIGUOUS'),      # node
        np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),      # infectious_timer
        np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),      # incubation_timer
        np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),      # immunity
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),    # age
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),    # expected_lifespan
        np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # infectious_count
        np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # incubating_count
        np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # susceptible_count
        np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),     # recovered_count
        np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS')      # dead_count
    ]
except Exception as ex:
    print( f"Failed to load libages.so." )

def init( model ):
    model.nodes.add_vector_property("S", model.params.ticks, dtype=np.uint32)
    model.nodes.add_vector_property("E", model.params.ticks, dtype=np.uint32)
    model.nodes.add_vector_property("I", model.params.ticks, dtype=np.uint32)
    model.nodes.add_vector_property("R", model.params.ticks, dtype=np.uint32)
    model.nodes.add_vector_property("D", model.params.ticks, dtype=np.uint32)


def update_ages( model, tick ):
    """
    lib.update_ages(
            ctypes.c_int64(model.population.count),
            model.population.age,
        )
    """
    S = np.zeros( len( model.nodes.S), dtype=np.uint32 )
    E = np.zeros( len( model.nodes.S), dtype=np.uint32 )
    I = np.zeros( len( model.nodes.S), dtype=np.uint32 )
    R = np.zeros( len( model.nodes.S), dtype=np.uint32 )
    D = np.zeros( len( model.nodes.S), dtype=np.uint32 )
    lib.update_ages_and_report(
            ctypes.c_int64(model.population.count),
            len(model.nodes.nn_nodes),
            model.population.age,
            model.population.nodeid,
            model.population.itimer,
            model.population.etimer,
            model.population.susceptibility,
            model.population.age,
            model.population.dod,
            S,
            E,
            I,
            R,
            D
        )
    model.nodes.S[:,tick]=S
    model.nodes.E[:,tick]=E
    model.nodes.I[:,tick]=I
    model.nodes.R[:,tick]=R
    model.nodes.D[:,tick]=D

