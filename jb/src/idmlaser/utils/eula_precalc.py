import numpy as np
from collections import defaultdict
from sparklines import sparklines
import sys
sys.path.append( "." )
import demographics_settings as settings
import pdb

# This file takes a eula_pop.csv file with initial populations by NODE and age
# and generates a CSV file with aggregate populations by node over time.

# Define Gompertz-Makeham parameters
makeham_parameter = 0.008
gompertz_parameter = 0.04
age_bins = np.arange(0, 102)
probability_of_dying = 2.74e-6 * ( makeham_parameter + np.exp(gompertz_parameter * (age_bins - age_bins[0])) )
#print( f"probability_of_dying = {probability_of_dying}" )
print( "probability_of_dying=" )
print( sparklines( probability_of_dying ) )

eula = defaultdict(lambda: defaultdict(int))

def init():
    global eula
    header_row = np.genfromtxt(settings.eula_file, delimiter=',', dtype=str, max_rows=1)

    print( f"Loading {settings.eula_file}." )
    # Load the remaining data as numerical values, skipping the header row
    data = np.genfromtxt(settings.eula_file, delimiter=',', dtype=float, skip_header=1)

    for row in data:
        node = int(row[0])
        age = int(float(row[1]))  # Convert string to float and then to int
        total = int(row[2])
        if node not in eula:
            eula[node] = {}
        eula[node][age] = total

init()

with open( sys.argv[1], "w" ) as output_file:
    output_file.write( "t,node,pop\n" )

    # Calculate the expected deaths and new population for next 20 years...
    for t in range(20*365):
        #print( f"Init pop for node {node} = {eula[node]}." )
        for node in eula:
            expected_deaths = 0
            for age in eula[node]:
                count = eula[node][age]
                if count>0:
                    expected_deaths = np.random.binomial(count, probability_of_dying[age])
                eula[node][age]-=expected_deaths 
            node_time_pop = sum(eula[node].values())
            #print( f"t={t},node={node},pop={node_time_pop}" )
            output_file.write( f"{t},{node},{node_time_pop}\n" )
            #print( f"node={node},age={age},t={t},pop={eula[node][age]}" )

