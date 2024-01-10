import csv
import numpy as np
from sparklines import sparklines
import settings

write_report = True

def init():
    # Create a CSV file for reporting
    csvfile = open( settings.report_filename, 'w', newline='') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'Recovered'])
    return csvwriter

def write_timestep_report( csvwriter, timestep, infected_counts, susceptible_counts, recovered_counts ):
    # This function is model agnostic
    infecteds = np.array([infected_counts[key] for key in sorted(infected_counts.keys(), reverse=True)])
    total = {key: susceptible_counts.get(key, 0) + infected_counts.get(key, 0) + recovered_counts.get(key, 0) for key in susceptible_counts.keys()}
    totals = np.array([total[key] for key in sorted(total.keys(), reverse=True)])
    prev = infecteds/totals
    print( f"T={timestep}" )
    print( list( sparklines( prev ) ) )
    # Write the counts to the CSV file
    #print( f"T={timestep},\nS={susceptible_counts},\nI={infected_counts},\nR={recovered_counts}" )
    if write_report:
        for node in settings.nodes:
            csvwriter.writerow([timestep,
                node,
                susceptible_counts[node] if node in susceptible_counts else 0,
                infected_counts[node] if node in infected_counts else 0,
                recovered_counts[node] if node in recovered_counts else 0,
                ]
            )

