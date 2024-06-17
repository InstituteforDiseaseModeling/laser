import csv
import numpy as np
import socket
from sparklines import sparklines
import time
import os
import settings
import demographics_settings

csv_report = True
binary_report = False
write_report = True
publish_report = False
new_infections = np.zeros(len(demographics_settings.nodes), dtype=np.uint32)
wtr_time = 0

# Configuration for the socket server
HOST = 'localhost'  # Use 'localhost' for local testing
PORT = 65432        # Port to bind the server to

client_sock = None
binary_data_accumulator = []

# Function to send CSV data over a socket
def send_csv_data(socket_conn, data):
    csvwriter = csv.writer(socket_conn.makefile('w', newline=''), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in data:
        csvwriter.writerow(row)

def is_running_in_notebook():
    return 'JUPYTERHUB_API_TOKEN' in os.environ

def init():
    csvwriter = None
    if write_report:
        # Create a CSV file for reporting
        # 2MB -> 396s. 8MB -> bad. 4MB -> 382 (1 time), 1MB -> 380, 0.5MB -> 480
        if is_running_in_notebook:
            csvfile = open( settings.report_filename, 'w', newline='' ) 
        else:
            csvfile = open( settings.report_filename, 'w', newline='', buffering=int(1024*1024))  
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'New Infections', 'Recovered', 'Births'])
    if publish_report:
        global client_sock
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect((HOST, PORT))
    return csvwriter

def write_timestep_report( csvwriter, timestep, infected_counts, susceptible_counts, recovered_counts, new_births ):
    wtr_start = time.time()
    # This function is model agnostic
    def show_sparklines():
        infecteds = np.array([infected_counts[key] for key in sorted(infected_counts.keys(), reverse=True)])
        total = {key: susceptible_counts.get(key, 0) + infected_counts.get(key, 0) + recovered_counts.get(key, 0) for key in susceptible_counts.keys()}
        totals = np.array([total[key] for key in sorted(total.keys(), reverse=True)])
        prev = infecteds/totals
        print( list( sparklines( prev ) ) )
    print( f"T={timestep}" )
    if not os.getenv( "HEADLESS" ):
        show_sparklines()
    # Write the counts to the CSV file
    #print( f"T={timestep},\nS={susceptible_counts},\nI={infected_counts},\nR={recovered_counts}" )
    if write_report and timestep >= settings.report_start:
        if csv_report:
            #print( "Writing CSV report for timestep." )
            for node in demographics_settings.nodes:
                csvwriter.writerow([timestep,
                    node,
                    susceptible_counts[node] if node in susceptible_counts else 0,
                    infected_counts[node] if node in infected_counts else 0,
                    new_infections[node],
                    recovered_counts[node] if node in recovered_counts else 0,
                    new_births[node],
                    ]
                )
        if binary_report:
            for node in demographics_settings.nodes:
                binary_data_accumulator.append(
                    [timestep,
                        node,
                        susceptible_counts[node] if node in susceptible_counts else 0,
                        infected_counts[node] if node in infected_counts else 0,
                        new_infections[node],
                        recovered_counts[node] if node in recovered_counts else 0,
                        new_births[node] if node in new_births else 0,
                    ]
                )

    if publish_report:
        data = []
        for node in demographics_settings.nodes:
            row = [
                timestep,
                node,
                susceptible_counts[node] if node in susceptible_counts else 0,
                infected_counts[node] if node in infected_counts else 0,
                new_infections[node],
                recovered_counts[node] if node in recovered_counts else 0,
                new_births[node] if node in new_births else 0,
            ]
            data.append(row)
        send_csv_data( client_sock, data )
    global wtr_time
    wtr_time += time.time() - wtr_start

def stop():
    if binary_report:
        # Convert accumulated data to a NumPy array
        data_array = np.array(binary_data_accumulator)

        # Save the array to a .npy file
        np.save('simulation_output.npy', data_array)

    if csv_report:
        csvwriter.flush()
        csvwriter.close()
