import numpy as np
from collections import defaultdict
import settings
import ctypes
import pdb

update_ages_lib = ctypes.CDLL('./update_ages.so')
update_ages_lib.progress_natural_mortality_binned.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), # eula
    ctypes.c_size_t,  # num_nodes
    ctypes.c_size_t,  # num_age_bins
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # probs
    ctypes.c_size_t,  # timesteps_elapsed
]

eula = defaultdict(lambda: defaultdict(int))

# Define Gompertz-Makeham parameters
makeham_parameter = 0.01
gompertz_parameter = 0.05
age_bins = np.arange(settings.eula_age, 102)
probability_of_dying = 2.74e-6 * ( makeham_parameter + np.exp(gompertz_parameter * (age_bins - age_bins[0])) )
#print( f"probability_of_dying = {probability_of_dying}" )
fits = np.load(setings.eula_pop_fits, allow_pickle=True).item()
def calculate_y(x, m, b):
    return int(m * x + b)

timestep_abs = 0

# call out to c function for this counting
def count_by_node_and_age( nodes, ages ):
    print( "Counting eulas by node and age; This is slow for now." )
    counts = defaultdict(lambda: defaultdict(int))
    for node_id, age in zip( nodes, ages ):
        age_bin = int(age)
        #age_bin = 44 # you can test out sticking everyone in a single bin
        counts[node_id][age_bin] += 1
    return counts

def init():
    global eula
    """
    header_row = np.genfromtxt(settings.eula_file, delimiter=',', dtype=str, max_rows=1)

    # Load the remaining data as numerical values, skipping the header row
    data = np.genfromtxt(settings.eula_file, delimiter=',', dtype=float, skip_header=1)

    for row in data:
        node = int(row[0])
        age = int(float(row[1]))  # Convert string to float and then to int
        total = int(row[2])
        if node not in eula:
            eula[node] = {}
        eula[node][age] = total
    """
    for node in range(settings.num_nodes):
        m, b = fits[node]
        pop = calculate_y(0, m, b)
        print( f"Setting pop for node {node} to {pop}." )
        eula[node][44] = pop
    

def progress_natural_mortality( timesteps ):
    global timestep_abs 
    timestep_abs += timesteps
    def python():
        def get_simple_death_rate( age_bin_in_yrs ): # This obviously needs to be done once and then returned from a lookup table.
            # Calculate the probability of dying using the Gompertz-Makeham distribution
            # Gompertz-Makeham distribution: hazard = makeham_parameter + exp(gompertz_parameter * age)
            try:
                return probability_of_dying#[ age_bin_in_yrs-settings.eula_age ]
            except Exception as ex:
                pdb.set_trace()

        return_deaths = defaultdict(int)
        for node in eula:
            expected_deaths = np.zeros(102-settings.eula_age).astype(np.int32)

            counts = np.zeros(102-settings.eula_age)

            # Update the array with the count values from the dictionary
            for key, value in eula[node].items():
                index = key - settings.eula_age  # Calculate the index based on the key
                counts[index] = int(value)

            for _ in range( timesteps ): # can't believe I have to loop this
                #count = list(eula[node].values()) # array, why doesn't this start at right age?
                prob = probability_of_dying  # array
                pdb.set_trace()
                if len(counts)!=len(prob):
                    pdb.set_trace()
                    raise ValueError( f"number of age bins in count={len(count)}, but number of age bins in prob={len(prob)} for node {node}." )
                expected_deaths += np.random.binomial(counts, prob)
            for age in eula[node]:
                eula[node][age] -= expected_deaths[age-settings.eula_age] # round(count * (1-))
            return_deaths[node] = sum(expected_deaths)

        """
        for node, age_bins_counts in eula.items():
            for age_bin, count in age_bins_counts.items():
                # Reduce count by 0.1%
                if eula[node][age_bin] > 0:
                    from scipy.stats import poisson, binom
                    prob = get_simple_death_rate( age_bin ) 
                    expected_deaths = sum(np.random.binomial(eula[node][age_bin], prob) for _ in range(timesteps))
                    if expected_deaths > 0:
                        #print( f"Killing off {expected_deaths} in node {node} and age_bin {age_bin} from existing population {eula[node][age_bin]} from prob {prob}." )
                        eula[node][age_bin] -= expected_deaths # round(count * (1-))
                        if eula[node][age_bin] == 0:
                            print( f"EULA bin for node {node} and age {age_bin} is now 0." )
                            #pdb.set_trace()
                    return_deaths[node] += expected_deaths 
        """
        return return_deaths
    def c():
        rows = len(eula)
        cols = len(next(iter(eula.values())))
        array = [[eula[row][col] for col in range(cols)] for row in range(rows)]

        # Convert 2D array to a flat array
        flat_array = np.array([elem for sublist in array for elem in sublist]).astype( np.int32 )

        return update_ages_lib.progress_natural_mortality_binned(
            # TBD: SORT
            flat_array, # sorted values as an array
            settings.num_nodes,
            cols,
            np.array( probability_of_dying ).astype( np.float32 ),
            timestep )

    def from_lut():
        # Calculate y values using the fit parameters and x values

        for node in range(settings.num_nodes):
            m, b = fits[node]
            pop = calculate_y(timestep_abs, m, b)
            eula[node][44] = pop

    from_lut()
    #python()

def get_recovereds_by_node():   
    summary = {}
    for node in eula:
        summary[ node ] = sum( eula[node].values() )
    return summary
