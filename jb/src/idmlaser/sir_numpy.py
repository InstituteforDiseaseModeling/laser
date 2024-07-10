import random
import csv
import numpy as np
import pandas as pd # for births.csv
import concurrent.futures
from functools import partial
import pdb

import settings
from . import report
from .model_numpy import eula
from .sir_sql import get_beta_samples

def add_expansion_slots( columns, num_slots=settings.expansion_slots ):
    """
    Adds 'expansion slots' to the agent population for future births.

    This function prepends a specified number of "expansion slots" to the existing
    agent population columns. Each expansion slot represents an agent to be born
    later and is initialized with default values. The function ensures that the
    new slots are contiguous in memory, allowing efficient management of the agent
    population as agents are born and the start index of the population array is
    decremented.

    Args:
        columns (dict): A dictionary where keys are column headers and values are
                        NumPy arrays representing the agent population attributes.
        num_slots (int, optional): The number of expansion slots to add. Defaults
                                   to settings.expansion_slots.

    Returns:
        None: The function modifies the 'columns' dictionary in place by appending
              new expansion slots to each column.

    Example:
        columns = {
            'id': np.array([1, 2, 3]),
            'node': np.array([0, 1, 0]),
            'age': np.array([25.0, 30.0, 22.0]),
            'infected': np.array([False, True, False]),
            'infection_timer': np.array([0, 5, 0]),
            'incubation_timer': np.array([0, 3, 0]),
            'immunity': np.array([True, False, True]),
            'immunity_timer': np.array([120.0, 0.0, 60.0]),
            'expected_lifespan': np.array([80.0, 75.0, 85.0])
        }
        add_expansion_slots(columns)

    Notes:
        - The function initializes new agents with default values such as -1 for
          'node' and 'age', False for 'infected', and specified values for other
          attributes. The goal is that as few properties as possible have to be set
          upon birth, but there is still a way to check that an agent is unborn
          (e.g., age=-1, node=-1).
        - It also updates the 'settings.nodes' and 'settings.num_nodes' based on
          the unique nodes in the 'columns'. This may be deprecated soon.

    """

    num_slots = int(num_slots)
    print( f"Adding {num_slots} expansion slots for future babies." )
    new_ids = [ x for x in range( num_slots ) ]
    new_nodes = np.ones( num_slots, dtype=np.int32 )*-1
    new_ages = np.ones( num_slots, dtype=np.float32 )*-1
    new_infected = np.zeros( num_slots, dtype=bool )

    #new_immunity = np.zeros( num_slots, dtype=bool )
    ew_immunity_timer = np.zeros( num_slots ).astype( np.float32 )

    # maternal immunity: not working right?!?!
    new_immunity = np.ones( num_slots, dtype=bool ) # not stable
    new_immunity_timer = np.ones( num_slots ).astype( np.float32 )*120 # stable

    new_infection_timer = np.zeros( num_slots ).astype( np.float32 )
    new_incubation_timer = np.zeros( num_slots ).astype( np.float32 )
    lifespan_samples = get_beta_samples( num_slots )
    new_expected_lifespan = np.array( lifespan_samples ).astype( dtype=np.float32 )

    settings.nodes = [ node for node in np.unique(columns['node']) ]
    settings.num_nodes = len(settings.nodes)
    #print( f"[sir_numpy] Nodes={settings.num_nodes}" )
    # Now 'columns' is a dictionary where keys are column headers and values are NumPy arrays

    columns['id'] = np.concatenate((columns['id'], new_ids))
    columns['node'] = np.concatenate((columns['node'], new_nodes)).astype( np.int32 )
    columns['age'] = np.concatenate((columns['age'], new_ages))
    columns['infected'] = np.concatenate((columns['infected'], new_infected))
    columns['infection_timer'] = np.concatenate((columns['infection_timer'], new_infection_timer))
    columns['incubation_timer'] = np.concatenate((columns['incubation_timer'], new_incubation_timer))
    columns['immunity'] = np.concatenate((columns['immunity'], new_immunity))
    columns['immunity_timer'] = np.concatenate((columns['immunity_timer'], new_immunity_timer))
    columns['expected_lifespan'] = np.concatenate((columns['expected_lifespan'], new_expected_lifespan))

def births_from_cbr_fast( node_pops_array, rate=30 ):
    """
    Calculate the number of births for each node based on the crude birth rate (CBR).

    This function computes the expected number of births for each node in a population
    using a given crude birth rate (CBR). The computation is performed in a vectorized
    manner for high performance.

    Args:
        node_pops_array (np.ndarray): An array containing the population of each node.
        rate (float, optional): The crude birth rate per 1,000 individuals per year.
                                Defaults to 30.

    Returns:
        np.ndarray: An array containing the number of new births for each node.

    Example:
        node_pops = np.array([1000, 2000, 1500])
        births = births_from_cbr_fast(node_pops, rate=30)
        print(births)  # Array of new births for each node.

    Notes:
        - The function uses a fertility interval defined in `settings.fertility_interval`
          to adjust the birth rate.
        - The crude birth rate (CBR) is scaled by the node populations and the number
          of days in a year (365).
        - Poisson-distributed random numbers are generated to simulate the number of new
          births, reflecting the natural variation in birth events.

    """

    # Compute the cbr_node for all nodes in a vectorized manner
    cbr_node_array = settings.fertility_interval * rate * (node_pops_array / 1000.0) / 365.0

    # Generate Poisson-distributed random numbers for all nodes in a vectorized manner
    new_babies_array = np.random.poisson(cbr_node_array)
    return new_babies_array 
  
