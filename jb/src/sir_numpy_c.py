import random
import csv
import numpy as np
import pandas as pd
import concurrent.futures
from functools import partial
import ctypes
import gzip
import pdb
import time
import gc

import settings
import demographics_settings
import report
#from model_sql import eula
from model_numpy import eula
import sir_numpy

from collections import defaultdict 
unborn_end_idx = 0
dynamic_eula_idx = None
ninemo_tracker_idx = None
inf_sus_idx = None
infection_queue_map = defaultdict(list)
incubation_queue_map = defaultdict(list)
recovereds = 0
infecteds = 0

s_to_i_swap_time = 0
i_to_r_swap_time = 0
attraction_probs = None
cbrs = None
low_infected_idx = 0
high_infected_idx = None
cr_time = 0
pi_time = 0
eula_reco_time = 0
birth_time = 0
funeral_time = 0
age_time = 0

# optional function to dump data to disk at any point. A sort-of serialization.
def dump():
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv('temp.csv', index=False)

def load_cbrs():
    # Read the CSV file into a DataFrame
    df = pd.read_csv( demographics_settings.cbr_file )

    # Initialize an empty dictionary to store the data
    cbrs_dict = {}

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Get the values from the current row
        elapsed_year = row['Elapsed_Years']
        node_id = row['ID']
        cbr = row['CBR']

        # If the year is not already in the dictionary, create a new dictionary for that year
        if elapsed_year not in cbrs_dict:
            cbrs_dict[elapsed_year] = []
        
        # If the node_id is not already in the dictionary for the current year, add it
        cbrs_dict[elapsed_year].append( cbr ) # is this guaranteed right order?

    return cbrs_dict

# Load the shared library
update_ages_lib = ctypes.CDLL('./update_ages.so')
# Define the function signature
update_ages_lib.update_ages.argtypes = [
    ctypes.c_size_t,  # start_idx
    ctypes.c_size_t,  # stop_idx
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
]
update_ages_lib.progress_infections.restype = ctypes.c_int
update_ages_lib.progress_infections.argtypes = [
    ctypes.c_size_t,  # starting index
    ctypes.c_size_t,  # ending index
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # infection_timer
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # incubation_timer
    np.ctypeslib.ndpointer(dtype=bool, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS'),  # immunity_timer
    np.ctypeslib.ndpointer(dtype=bool, flags='C_CONTIGUOUS'),  # immunity
    #np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # recovered idxs out
]
update_ages_lib.progress_immunities.argtypes = [
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # starting index
    np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS'),  # immunity_timer
    np.ctypeslib.ndpointer(dtype=bool, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # node
]
update_ages_lib.calculate_new_infections.argtypes = [
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # starting index
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # nodes
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # incubation_timer
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # inf_counts
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # sus_counts
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # tot_counts
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'),  # new_infections
    ctypes.c_float, # base_inf
]
update_ages_lib.handle_new_infections.argtypes = [
    ctypes.c_uint32, # num_agents
    ctypes.c_uint32, # node
    ctypes.c_size_t,  # starting index
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # nodes
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # incubation_timer
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # infection_timer
    ctypes.c_int, # num_new_infections
    #np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # new infected ids
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # new infected ids
    ctypes.c_int, # num_new_infections
    #np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # array of no. susceptibles by node
]
update_ages_lib.handle_new_infections_mp.argtypes = [
    ctypes.c_uint32, # num_agents
    ctypes.c_size_t,  # starting index
    ctypes.c_size_t,  # num_nodes
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # nodes
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # incubation_timer
    np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), # infection_timer
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # array of no. new infections to create by node
    #np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # new infected ids
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # array of no. susceptibles by node
]
update_ages_lib.migrate.argtypes = [
    ctypes.c_int, # start_idx
    ctypes.c_int, # stop_idx
    np.ctypeslib.ndpointer(dtype=bool, flags="C_CONTIGUOUS"),  # infected
    np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),  # incubation_timer
    np.ctypeslib.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),  # data_node
    np.ctypeslib.ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),  # data_home_node
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # attraction_probs
    ctypes.c_double,  # migration_fraction
    ctypes.c_int,  # num_locations
]
update_ages_lib.collect_report.argtypes = [
    ctypes.c_uint32, # num_agents
    ctypes.c_size_t,  # starting index
    ctypes.c_size_t,  # eula index
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # nodes
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # infected
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # age
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # expected_lifespan
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # infection_count_out
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # susceptible_count_out
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # recovered_count_out
]
update_ages_lib.campaign.argtypes = [
    ctypes.c_int32, # num_agents
    ctypes.c_size_t,  # starting index
    ctypes.c_float, # coverage
    ctypes.c_int32, # campaign_node
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS'), # immunity_timer
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # age
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # node
]
update_ages_lib.ria.argtypes = [
    ctypes.c_int32, # num_agents
    ctypes.c_size_t,  # starting index
    ctypes.c_float, # coverage
    ctypes.c_int32, # campaign_node
    np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS'),  # immunity
    np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS'), # immunity_timer
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # age
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # node
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # vaccinated indices
    
]
update_ages_lib.reconstitute.argtypes = [
    ctypes.c_int32, # num_new_babies
    ctypes.c_size_t,  # starting index
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # new_nodes
    np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS'), # node array
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # age
]

def do_due_tasks( ctx, timestep ):
    return sir_numpy.do_due_tasks( ctx, timestep )

def load( pop_file ):
    """
    Load population from csv file as np arrays. Each property column is an np array.
    """
    with gzip.open( pop_file ) as fp:
        # Load the entire CSV file into a NumPy array
        header_row = np.genfromtxt(fp, delimiter=',', dtype=str, max_rows=1)

        # Load the remaining data as numerical values, skipping the header row
        data = np.genfromtxt(fp, delimiter=',', dtype=float, skip_header=1)

        # Extract headers from the header row
        headers = header_row

    settings.pop = len(data) # -unborn_end_idx 
    print( f"Population={settings.pop}" )

    unborn = {header: [] for i, header in enumerate(headers)}
    global unborn_end_idx 
    sir_numpy.add_expansion_slots( unborn, num_slots=settings.expansion_slots )
    unborn_end_idx = int(settings.expansion_slots)
    low_infected_idx = unborn_end_idx 

    data = {header: data[:, i] for i, header in enumerate(headers)}
    # Load each column into a separate NumPy array
    #columns = {header: data[:, i] for i, header in enumerate(headers)}
    # TBD: data['id'] has to be handled and started from unborn_end_idx 
    data['id'] = np.concatenate( ( unborn['id'], data['id'] + len(unborn['id'])-1) )
    data['infected'] = np.concatenate( ( unborn['infected'], data['infected'] ) ).astype(bool)
    data['immunity'] = np.concatenate( ( unborn['immunity'], data['immunity'] ) ).astype(bool)
    data['node'] = np.concatenate( [ unborn['node'], data['node'] ] ).astype(np.uint32)
    data['infection_timer'] = np.concatenate( [ unborn['infection_timer'], data['infection_timer'] ] ).astype(np.uint8)
    data['incubation_timer'] = np.concatenate( [ unborn['incubation_timer'], data['incubation_timer'] ] ).astype(np.uint8)
    data['immunity_timer'] = np.concatenate( [ unborn['immunity_timer'], data['immunity_timer'] ] ).astype(np.int8)
    data['age'] = np.concatenate( [ unborn['age'], data['age'] ] ).astype(np.float32)
    data['expected_lifespan'] = np.concatenate( [ unborn['expected_lifespan'], data['expected_lifespan'] ] ).astype(np.float32)
    data['home_node'] = np.ones( len(data['id'] ) ).astype(np.int32)*-1

    def clear_init_prev():
        data['incubation_timer'] = np.zeros( len( data['incubation_timer'] ) ).astype( np.uint8 )
        data['infection_timer'] = np.zeros( len( data['incubation_timer'] ) ).astype( np.uint8 )
        data['infected'] = np.zeros( len( data['incubation_timer'] ) ).astype( bool )
    clear_init_prev()

    settings.nodes = [ node for node in np.unique(data['node']) ]
    settings.nodes.pop(-1)
    settings.num_nodes = len(settings.nodes)
    print( f"Nodes={settings.num_nodes}" )
    global dynamic_eula_idx, ninemo_tracker_idx, inf_sus_idx, high_infected_idx 
    dynamic_eula_idx = len(data['id'])-1
    ninemo_tracker_idx = dynamic_eula_idx 
    inf_sus_idx = dynamic_eula_idx 
    high_infected_idx = dynamic_eula_idx 

    # Now 'columns' is a dictionary where keys are column headers and values are NumPy arrays

    def load_attraction_probs():
        probabilities = []
        with open(settings.attraction_probs_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                probabilities.append([float(prob) for prob in row])
            return np.array(probabilities)
        return probabilities

    global attraction_probs 
    if demographics_settings.num_nodes > 1 and settings.migration_fraction > 0:
        attraction_probs = load_attraction_probs()
    return data

def initialize_database():
    return load( demographics_settings.pop_file )

def eula_init( df, age_threshold_yrs = 5, eula_strategy=None ):
    eula.init()
    return df

def swap_to_dynamic_eula( data, individual_idx ):
    global dynamic_eula_idx 
    #global i_to_r_swap_time 
    #start_time = time.time()
    """
    if individual_idx > dynamic_eula_idx:
        raise ValueError( f"It should not be possible for the individual idx {individual_idx} to be already in the eula region {dynamic_eula_idx}." )
    """
    #print( f"swapping newly recovered individual at {individual_idx} with dynamic eula idx {dynamic_eula_idx}." )
    #individual_idx = np.where( data['id'] == individual_id  )
    #print( f"EULA: idx={dynamic_eula_idx}" )
    #print( f"CUR: idx={individual_idx}" )
    for col in data.keys():
        # Store eula-1 values in temp
        elem = data[ col ][ dynamic_eula_idx ]
        #print( f"EULA: {col}={elem}" )
        #print( f"CUR: {col}={data[ col ][ individual_idx ]}" )
        # Write newly recovered values to eula-1
        data[ col ][ dynamic_eula_idx ] = data[ col ][ individual_idx ]
        # Write temp values to current
        data[ col ][ individual_idx ] = elem
    dynamic_eula_idx -= 1
    #print( f"dynamic_eula_idx = {dynamic_eula_idx}" )
    global recovereds
    recovereds += 1
    #print( f"recovereds = {recovereds}" )
    """
    if recovereds > infecteds:
        raise ValueError( f"ERROR: How do we have more recovereds {recovereds} than infecteds {infecteds}." )
    global inf_sus_idx
    if dynamic_eula_idx < inf_sus_idx:
        raise ValueError( f"dynamic_eula_idx ({dynamic_eula_idx}) should never be less than inf_sus_idx ({inf_sus_idx})." )
    #print( f"dynamic_eula_idx decremented to {dynamic_eula_idx}" )
    """
    #i_to_r_swap_time += time.time() - start_time

def swap_to_dynamic_si( data, individual_idx ):
    global infecteds
    global inf_sus_idx
    start_time = time.time()
    
    #print( f"swapping newly infected individual at {individual_idx} with s_i idx {inf_sus_idx}." )
    if individual_idx > inf_sus_idx:
        raise ValueError( f"It should not be possible for the newly infected individual idx {individual_idx} to be already in the infected region (or greater) {inf_sus_idx}." )
        pdb.set_trace()
    # will consolidate with above
    for col in data.keys():
        # Store eula-1 values in temp
        elem = data[ col ][ inf_sus_idx ]
        #print( f"EULA: {col}={elem}" )
        #print( f"CUR: {col}={data[ col ][ individual_idx ]}" )
        # Write newly recovered values to eula-1
        data[ col ][ inf_sus_idx ] = data[ col ][ individual_idx ]
        # Write temp values to current
        data[ col ][ individual_idx ] = elem

    inf_sus_idx -= 1
    #print( f"infecteds = {infecteds}" )
    #print( f"inf_sus_idx decremented to {inf_sus_idx}" )
    global s_to_i_swap_time
    s_to_i_swap_time += time.time() - start_time

def collect_report( data ):
    """
    Report data to file for a given timestep.
    """
    infected_counts_raw = np.zeros( settings.num_nodes ).astype( np.uint32 )
    susceptible_counts_raw = np.zeros( settings.num_nodes ).astype( np.uint32 )
    recovered_counts_raw = np.zeros( settings.num_nodes ).astype( np.uint32 )

    cr_start = time.time()
    #print( f"unborn_end_idx={unborn_end_idx}, dynamic_eula_idx={dynamic_eula_idx}." )
    update_ages_lib.collect_report(
            len( data['node'] ),
            unborn_end_idx,
            dynamic_eula_idx,
            data['node'],
            data['infected'],
            data['immunity'],
            data['age'],
            data['expected_lifespan'],
            infected_counts_raw,
            susceptible_counts_raw,
            recovered_counts_raw
    )
    global cr_time
    cr_time += time.time() - cr_start

    eula_reco_start = time.time()
    #print( f"infected_counts_raw = {infected_counts_raw}" )
    susceptible_counts = dict(zip(settings.nodes, susceptible_counts_raw))
    infected_counts = dict(zip(settings.nodes, infected_counts_raw))

    recovered_counts = recovered_counts_raw + np.array( eula.get_recovereds_by_node_np() )

    totals = susceptible_counts_raw + infected_counts_raw + recovered_counts
    recovered_counts = dict(zip(settings.nodes, recovered_counts))

    #recovered_counts = dict(zip(settings.nodes, recovered_counts_raw))
    #print( f"Reporting back SIR counts of\n{susceptible_counts},\n{infected_counts}, and\n{recovered_counts}." )

    #recovered_counts_eula = eula.get_recovereds_by_node()
    #print( f"Number of recovereds in London now = {eula.eula_dict[507][44]}." )
    """
    for node in eula.eula_dict:
        #if node not in recovered_counts:
        #    recovered_counts[ node ] = 0
        recovered_counts[ node ] += recovered_counts_eula[node]
    """
    global eula_reco_time
    eula_reco_time += time.time() - eula_reco_start

    return infected_counts, susceptible_counts, recovered_counts, totals
    

def update_ages( data, totals, timestep ):
    def update_ages_c( ages ):
        update_ages_lib.update_ages(
            unborn_end_idx,
            dynamic_eula_idx,
            ages
        )
        return ages
    def update_ages_np( ages ):
        one_day = 1.0 / 365.0
        ages[unborn_end_idx:dynamic_eula_idx+1][ages[unborn_end_idx:dynamic_eula_idx+1] >= 0] += one_day
        return ages

    import numba as nb
    @nb.njit
    def update_ages_numba( ages ):
    #def update_ages(ages, unborn_end_idx, dynamic_eula_idx):
        one_day = 1.0 / 365.0
        for i in range(unborn_end_idx, dynamic_eula_idx+1):
            if ages[i] >= 0:
                ages[i] += one_day

    if not data:
        raise ValueError( "update_ages called with null data variable." )

    #update_ages_np( data['age'] )
    age_start_time = time.time()
    update_ages_c( data['age'] )
    global age_time
    age_time += time.time() - age_start_time
    #update_ages_numba( data['age'] )

    global unborn_end_idx
    def births( data, interval ):
        labor_start_time = time.time()
        #import sir_numpy

        num_new_babies_by_node = sir_numpy.births_from_cbr_fast( totals, rate=settings.cbr )

        """
        global cbrs
        if not cbrs:
            cbrs = load_cbrs()
        num_new_babies_by_node = sir_numpy.births_from_cbr_var( totals, rate=cbrs[timestep//365] )
        """

        #num_new_babies_by_node = sir_numpy.births_from_lorton_algo( timestep )

        #keys = np.array(list(num_new_babies_by_node.keys()))
        #values = np.array(list(num_new_babies_by_node.values()))
        # Create an array of indices (nodes)
        nodes = np.arange(len(num_new_babies_by_node))

        # Repeat the nodes according to the values in num_new_babies_by_node
        result_array = np.repeat(nodes, num_new_babies_by_node).astype( np.uint32 )
        #result_array = np.repeat(keys, values)
        #tot_new_babies = sum(num_new_babies_by_node.values())
        tot_new_babies = sum(num_new_babies_by_node)
        global unborn_end_idx
        if tot_new_babies > 0 :
            update_ages_lib.reconstitute(
                unborn_end_idx-1, 
                len(result_array),
                result_array,
                data['node'],
                data['age']
            )
        """
        # TBD: Schedule 9mo RIA for newborns; Need their ids.
        import sir_numpy
        for new_id in new_ids_out:
            sir_numpy.schedule_9mo_ria( new_id, 0, timestep=timestep )
        """
        """
        births_report = defaultdict(int)
        for node, babies in num_new_babies_by_node.items():
            if babies > 0:
                if node not in births_report:
                    births_report[node] = 0
                births_report[node] += babies
        """

        unborn_end_idx -= tot_new_babies # len( new_ids_out )
        #print( f"unborn_end_idx = {unborn_end_idx} after {tot_new_babies} new babies." )
        global birth_time
        birth_time += time.time() - labor_start_time
        #return births_report
        return num_new_babies_by_node

    def deaths( data, timestep_delta ):
        funeral_start_time = time.time()
        death_report = eula.progress_natural_mortality(timestep_delta)
        global funeral_time
        funeral_time += time.time() - funeral_start_time
        return death_report 

    #birth_report = {}
    birth_report = []
    death_report = {}
    if timestep % settings.fertility_interval == 0:
        birth_report = births( data, settings.fertility_interval )
    #print( f"births: {report}" )
    if timestep % settings.mortality_interval == 0:
        death_report = deaths( data, settings.mortality_interval )

    #print( f"Returning {birth_report}, and {death_report}" )
    return ( birth_report, death_report )

def progress_infections( data ):
    # Update infected agents
    # infection timer: decrement for each infected person
    # Would be nice to get indices (not ids) of newly recovereds...
    start_time = time.time()
    #recovered_idxs = np.zeros( num_infected ).astype( np.uint32 )
    global dynamic_eula_idx
    # infecteds should all be from inf_sus_idx (E/I) to dynamic_eula_idx (R)
    num_recovereds = update_ages_lib.progress_infections(
        unborn_end_idx,
        dynamic_eula_idx,
        data['infection_timer'],
        data['incubation_timer'],
        data['infected'],
        data['immunity_timer'],
        data['immunity'],
        None, # recovered_idxs
    ) # ctypes.byref(integers_ptr))
    global pi_time 
    pi_time += time.time() - start_time
    #print( f"{num_recovereds} recovered from their infections and are now immune." )
    #global i_to_r_swap_time 
    #start_time = time.time()
    """
    if num_recovereds > num_infected:
        print( f"ERROR: Somehow we got more recovered {num_recovereds} than infected {num_infected}!" )
        raise ValueError( f"ERROR: Somehow we got more recovered {num_recovereds} than infected {num_infected}!" )
    """
    """
    for rec_idx in range( num_recovereds ):
        recovered = recovered_idxs[rec_idx]
        if recovered > 0:
            if recovered < dynamic_eula_idx: # only swap if dynamic eula idx is greater than recovered idx. This was a bug I took a while to find. eula idx can cross the index while in queue
                swap_to_dynamic_eula( data, recovered )
    i_to_r_swap_time += time.time() - start_time
    """

    return data

# Update immune agents
def progress_immunities( data ):
    update_ages_lib.progress_immunities(unborn_end_idx, dynamic_eula_idx, data['immunity_timer'], data['immunity'], data['node'])
    return data

def calculate_new_infections( data, inf, sus, totals, timestep, **kwargs ):
    def cni_c():
        # Are inf and sus fractions or totals? fractions
        new_infections = np.zeros( len( inf ) ).astype( np.uint32 ) # return variable
        sorted_items = sorted(inf.items())
        # This code is when inf and sus are fractions; moving to counts
        #inf_np = np.array([np.float32(value) for _, value in sorted_items])
        #sus_np = np.array([np.float32(value) for value in sus.values()])
        # counts
        inf_np = np.array([value for _, value in sorted_items])
        sus_np = np.array([value for value in sus.values()])

        tot_np = np.array(totals).astype(np.uint32) # np.array([np.uint32(value) for value in totals.values()])

        sm = kwargs.get('seasonal_multiplier')
        inf_multiplier = max(0, 1 + sm * settings.infectivity_multiplier[ min((timestep%365) // 7, 51) ] )
        bi = kwargs.get('base_infectivity')
        #print( f"inf_multiplier = {inf_multiplier}" )
        update_ages_lib.calculate_new_infections(
                inf_sus_idx, # unborn_end_idx,
                dynamic_eula_idx,
                len( inf ),
                data['node'],
                data['incubation_timer'],
                # fractions
                #(inf_np*tot_np).astype( np.uint32 ),
                #(sus_np*tot_np).astype( np.uint32 ),
                # counts
                inf_np,
                sus_np,

                tot_np,
                new_infections,
                bi * inf_multiplier
            )
        #print( f"DEBUG: new_infections = {new_infections}." )
        return new_infections

    new_infections = cni_c()
    #import sir_numpy as py_model
    #new_infections = py_model.calculate_new_infections( data, inf, sus, totals )
    return new_infections 

def handle_transmission_by_node( data, new_infections, susceptible_counts, node=0 ):
    # print( f"DEBUG: New Infections: {new_infections}" )
    # print( f"DEBUG: susceptible_counts: {susceptible_counts}" )
    if new_infections[node]>susceptible_counts[node]:
        raise ValueError( f"ERROR: Asked for {new_infections[node]} new infections but only {susceptible_counts[node]} susceptibles exist in node {node}." )

    # Step 5: Update the infected flag for NEW infectees
    def handle_new_infections_c(new_infections):
        if new_infections > 1e6: # arbitrary "too large" value:
            print( "ERROR: new_infections value probably = -1. Ignore and continue." )
            return np.zeros(1).astype( np.uint32 )
        new_infection_idxs = np.zeros(new_infections).astype( np.uint32 )
        update_ages_lib.handle_new_infections(
                unborn_end_idx, # we waste a few cycles now coz the first block is immune from maternal immunity
                inf_sus_idx, # dynamic_eula_idx,
                node,
                data['node'],
                data['infected'],
                data['immunity'],
                data['incubation_timer'],
                data['infection_timer'],
                new_infections,
                new_infection_idxs,
                susceptible_counts[ node ]
            )

        #print( f"New Infection indexes = {new_infection_idxs} in node {node} at {timestep}." )
        #print( f"New Infection ids = {data['id'][new_infection_idxs]}." )
        return new_infection_idxs

    def handle_new_infections_np(new_infections):
        import sir_numpy as py_model
        return py_model.handle_transmission_by_node( data, new_infections, node )

    #print( f"new_infections at node {node} = {new_infections[node]}" )
    niis = handle_new_infections_c(new_infections[node])

    """
    global low_infected_idx, high_infected_idx
    for idx in niis:
        if idx < low_infected_idx:
            low_infected_idx = idx
        if idx > high_infected_idx:
            high_infected_idx = idx
    """
 
    return niis # data

def handle_transmission( data_in, new_infections_in, susceptible_counts ):
    # We want to do this in parallel;
    #print( f"DEBUG: New Infections: {new_infections_in}" )
    """
    htbn = partial( handle_transmission_by_node, data_in, new_infections_in, susceptible_counts )
    relevant_nodes = [ node_id for node_id in settings.nodes if new_infections_in[ node_id ] > 0 ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(htbn, relevant_nodes))
    """
    # new_infection_idxs = np.zeros(sum(new_infections_in)).astype( np.uint32 )
    update_ages_lib.handle_new_infections_mp(
    #update_ages_lib.handle_new_infections(
        unborn_end_idx, # we waste a few cycles now coz the first block is immune from maternal immunity
        #inf_sus_idx, # dynamic_eula_idx,
        dynamic_eula_idx,
        settings.num_nodes,
        data_in['node'],
        data_in['infected'],
        data_in['immunity'],
        data_in['incubation_timer'],
        data_in['infection_timer'],
        new_infections_in,
        #new_infection_idxs,
        np.array(list(susceptible_counts.values())).astype( np.uint32 )
    )

    # Segregate S from I: Swap new I's out of S group into I group
    """
    if len( results ) > 0:
        all_new_idxs = []
        for result in results:
            all_new_idxs.extend( result )
        for idx in sorted(all_new_idxs,reverse=True):
            if idx > 0:
                #print( f"New infection has age {data_in['age'][idx]}" )
                swap_to_dynamic_si( data_in, idx )
    """
    """
    for idx in sorted(new_infection_idxs,reverse=True):
        if idx > 0:
            swap_to_dynamic_si( data_in, idx )
    """
    #global infecteds
    #infecteds += len(new_infection_idxs)
    return data_in

def add_new_infections( data ):
    return data

def migrate( data, timestep, **kwargs ):
    # Migrate 1% of infecteds "downstream" every week; coz
    def return_home():
        # Find indices where home_node is greater than -1 and within the specified range
        #idx = data['home_node'] > -1 # works but slow
        low_idx = low_infected_idx
        high_idx = high_infected_idx
        idx = (data['home_node'][low_idx:high_idx+1] > -1)

        # Update node values where home_node is greater than -1 and within the specified range
        #data['node'][idx] = data['home_node'][idx]
        data['node'][low_idx:high_idx+1][idx] = data['home_node'][low_idx:high_idx+1][idx]

        # Set home_node to -1 for all indices where it was greater than -1 and within the specified range
        #data['home_node'][idx] = -1
        data['home_node'][low_idx:high_idx+1][idx] = -1

    #return_home() # this is expensive if we do it every time

    if timestep % settings.migration_interval == 0: # every week
        def select_destination(source_index, random_draw):
            return np.argmax(attraction_probs[source_index] > random_draw)

        """
        Code from demo for inspiration:
        source_location = random.randint(0,940)
        random_draw = np.random.rand()  # Random number between 0 and 1
        destination_location = select_destination(source_location, random_draw)
        city = all_cities[ destination_location ]
        """
        """
        #print( "Starting migration..." )
        # Let's migrate 1% of infected agents.
        # Select indices where 'infected' is True
        infected_indices = np.where((data['infected'] == True ) & ( data['incubation_timer']<=0 ))[0]

        # Calculate the number of individuals to select (1% of the total infected individuals)
        #num_to_select = int(len(infected_indices) * 0.01)
        mf = kwargs.get('migration_fraction')
        num_to_select = int(len(infected_indices) * mf)

        if num_to_select > 0:
            # Randomly select 1% of infected individuals
            selected_indices = np.random.choice(infected_indices, size=num_to_select, replace=False)

            # Get the selected individuals from the original data
            #selected_individuals = data[selected_indices]

            # Collect the source nodes in a list.
            source_nodes = data['node'][selected_indices]

            dest_nodes = []
            # Calculate destination nodes with looped calls to select_destination
            for src_node in source_nodes:
                random_draw = np.random.rand()  # Random number between 0 and 1
                destination_location = select_destination(src_node, random_draw)
                dest_nodes.append( destination_location )
            data['home_node'][selected_indices] = data['node'][selected_indices]
            data['node'][selected_indices] = np.array(dest_nodes)

        # Pass source ids and destinations to function?
        """
        update_ages_lib.migrate(
            unborn_end_idx,
            dynamic_eula_idx,
            data['infected'],
            data['incubation_timer'],
            data['node'],
            data['home_node'],
            attraction_probs,
            settings.migration_fraction,
            demographics_settings.num_nodes
        )
        #print( "Ending migration..." )

    # forced garbage collection is necessary due to something in handle_new_infections but done here so it's not done every timestep
    #gc.collect()
    return data

def distribute_interventions( data, timestep ):
    #import sir_numpy
    #return sir_numpy.distribute_interventions( ctx, timestep )
    def ria_9mo( coverage ):
        global ninemo_tracker_idx 
        vaxxed_idxs = np.zeros( ninemo_tracker_idx - unborn_end_idx ).astype( np.uint32 )
        new_idx = update_ages_lib.ria(
                unborn_end_idx,
                ninemo_tracker_idx,
                coverage,
                settings.campaign_node,
                data['immunity'],
                data['immunity_timer'],
                data['age'],
                data['node'],
                vaxxed_idxs
            )
        ninemo_tracker_idx = int(new_idx)
        idx = 0
        while vaxxed_idxs[ idx ] > 0:
            print( "Swapping previously S to R after RIA vax." )
            swap_to_dynamic_eula( data, vaxxed_idxs[ idx ] )
            idx += 1

    def campaign( coverage ):
        vaxxed = update_ages_lib.campaign(
                len(data['age']),
                unborn_end_idx,
                settings.campaign_coverage,
                settings.campaign_node,
                data['immunity'],
                data['immunity_timer'],
                data['age'].astype(np.float32),
                data['node']
            )
        print( f"{vaxxed} individuals vaccinated in node {settings.campaign_node}." )
    if timestep == settings.campaign_day:
        campaign(settings.campaign_coverage)
    if timestep % settings.ria_interval == 0:
        ria_9mo( settings.campaign_coverage )
    return data

def inject_cases( ctx, sus, import_cases=100, import_node=demographics_settings.num_nodes-1 ):
    import_dict = { import_node: import_cases }
    #import_dict = { import_node: int(0.1*sus[import_node]) }
    htbn = partial( handle_transmission_by_node, ctx, import_dict, susceptible_counts=sus, node=import_node )
    new_idxs = htbn()
    """
    new_idxs = handle_transmission(
        ctx, import_dict, susceptible_counts=sus, node=import_node
    )
    """
    """
    for idx in sorted(new_idxs,reverse=True):
        swap_to_dynamic_si( ctx, idx )
    """
    infecteds += len(new_idxs)

# Function to run the simulation for a given number of timesteps
def run_simulation(data, csvwriter, num_timesteps):
    currently_infectious, currently_sus, cur_reco  = collect_report( data )
    report.write_timestep_report( csvwriter, 0, currently_infectious, currently_sus, cur_reco )

    for timestep in range(1, num_timesteps + 1):
        data = update_ages( data )

        data = progress_infections( data )

        data = progress_immunities( data )

        new_infections = calculate_new_infections( data, currently_infectious, currently_sus )

        data = handle_transmission( data_in=data, new_infections_in=new_infections )

        data = add_new_infections( data )

        data = migrate( data, timestep )

        currently_infectious, currently_sus, cur_reco = collect_report( data )
        report.write_timestep_report( csvwriter, timestep, currently_infectious, currently_sus, cur_reco )


    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    data = initialize_database()

    # Create a CSV file for reporting
    csvfile = open('simulation_report.csv', 'w', newline='') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'Recovered'])

    # Run the simulation for 1000 timesteps
    run_simulation(data, csvwriter, num_timesteps=settings.duration )

