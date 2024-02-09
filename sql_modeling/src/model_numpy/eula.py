import numpy as np
from collections import defaultdict
import settings

eula = defaultdict(lambda: defaultdict(int))

# Define Gompertz-Makeham parameters
makeham_parameter = 0.01
gompertz_parameter = 0.05
age_bins = np.arange(15, 102)
probability_of_dying = 2.74e-6 * ( makeham_parameter + np.exp(gompertz_parameter * (age_bins - age_bins[0])) )

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
    header_row = np.genfromtxt(settings.eula_file, delimiter=',', dtype=str, max_rows=1)

    # Load the remaining data as numerical values, skipping the header row
    data = np.genfromtxt(settings.eula_file, delimiter=',', dtype=float, skip_header=1)

    # Extract headers from the header row
    headers = header_row

    # Load each column into a separate NumPy array
    columns = {header: data[:, i] for i, header in enumerate(headers)}
    columns['node'] = columns['node'].astype(np.uint32)
    columns['age'] = columns['age'].astype(np.float32)
    nodes = columns['node']
    ages = columns['age']

    eula = count_by_node_and_age( nodes, ages )

def progress_natural_mortality( timesteps ):
    def get_simple_death_rate( age_bin_in_yrs ):
        # This obviously needs to be done once and then returned from a lookup table.

        # Calculate the probability of dying using the Gompertz-Makeham distribution
        # Gompertz-Makeham distribution: hazard = makeham_parameter + exp(gompertz_parameter * age)
        try:
            return probability_of_dying[ age_bin_in_yrs-15 ] # hack 15 is eula age for now
        except Exception as ex:
            pdb.set_trace()

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

def get_recovereds_by_node():   
    summary = {}
    for node in eula:
        summary[ node ] = sum( eula[node].values() )
    return summary
