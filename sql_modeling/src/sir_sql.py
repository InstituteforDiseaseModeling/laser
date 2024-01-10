import sqlite3
import random
import csv
import concurrent.futures
import numpy as np # not for modeling
import pdb
import sys
import os

from settings import * # local file
import settings

settings.base_infectivity = 0.0001

# Globals! (not really)
conn = sqlite3.connect(":memory:")  # Use in-memory database for simplicity
cursor = conn.cursor() # db-specific
# use cursor as model data context; cf dataframe for polars/pandas
#conn = sqlite3.connect("simulation.db")  # Great for inspecting; presumably a bit slower

def get_node_ids():
    import numpy as np

    array = []
    for node in settings.nodes:
        array.extend( np.ones(node+1)*(node) )
    # Generate the array based on the specified conditions
    """
    array = np.concatenate([
        np.zeros(1),      # 0s
        np.ones(2),        # 1s
        2 * np.ones(3),   # 2s
        3 * np.ones(4),   # 3s
        4 * np.ones(5)    # 4s
    ])
    """

    array = np.tile(array, (pop + len(array) - 1) // len(array))[:pop]

    # Shuffle the array to randomize the order
    np.random.shuffle(array)

    # Convert the array to integers
    array = array.astype(int).tolist()

    # Print the first few elements as an example
    #print(array[:20])

    return array

# Function to initialize the SQLite database
def initialize_database( conn=None ):
    print( "Initializing pop NOT from file." )

    if not conn:
        conn = sqlite3.connect(":memory:")  # Use in-memory database for simplicity
    cursor = conn.cursor()

    # Create agents table
    cursor.execute('''
        CREATE TABLE agents (
            id INTEGER PRIMARY KEY,
            node INTEGER,
            age REAL,
            infected BOOLEAN,
            infection_timer INTEGER,
            incubation_timer INTEGER,
            immunity BOOLEAN,
            immunity_timer INTEGER
        )
    ''')
    # lots of index experiments going on here
    #cursor.execute( "CREATE INDEX node_idx ON agents(node)" )
    cursor.execute( "CREATE INDEX immunity_idx ON agents(immunity)" )
    cursor.execute( "CREATE INDEX immunity_node_idx ON agents(immunity,node)" )
    #cursor.execute( "CREATE INDEX infected_idx ON agents(infected)" )
    #cursor.execute( "CREATE INDEX idx_agents_node ON agents(id, node)" )
    #cursor.execute( "CREATE INDEX idx_agents_node_infected ON agents(node, infected)" )
    #cursor.execute( "CREATE INDEX idx_agents_node_immunity ON agents(node, immunity)" )
                

    # Insert 10,000 agents with random age and all initially uninfected
    #agents_data = [(i, random.randint(0, num_nodes-1), random.randint(0, 100), False, 0, 0, False, 0) for i in range(1, pop)]
    node_assignments = get_node_ids()
    agents_data = [(i, node_assignments[i], random.randint(0, 100), False, 0, 0, False, 0) for i in range(1, pop)]
    cursor.executemany('INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?)', agents_data)

    # Seed exactly 100 people to be infected in the first timestep
    # uniform distribution draws seem a bit clunky in SQLite. Probably better to use %
    cursor.execute( 'UPDATE agents SET infected = 1, infection_timer=CAST( 4+10*(RANDOM() + 9223372036854775808)/18446744073709551616 AS INTEGER ), incubation_timer=3 WHERE id IN (SELECT id FROM agents WHERE node=:big_node ORDER BY RANDOM() LIMIT 100)', { 'big_node': num_nodes-1 } )
    # Make everyone over some age perman-immune
    cursor.execute( "UPDATE agents SET immunity = 1, immunity_timer=-1 WHERE infected=0 AND age > 5*365" )

    conn.commit()

    return cursor

def collect_report( cursor ):
    #print( "Start report." ) # helps with visually sensing how long this takes.
    # Count agents in each state
    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE infected=0 AND immunity=0 GROUP BY node')
    # this seems slow and clunky
    
    susceptible_counts_db = cursor.fetchall()
    susceptible_counts = {values[0]: values[1] for idx, values in enumerate(susceptible_counts_db)}
    for node in nodes:
        if node not in susceptible_counts:
            susceptible_counts[node] = 0

    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE infected=1 GROUP BY node')
    infected_counts_db = cursor.fetchall()
    infected_counts = {values[0]: values[1] for idx, values in enumerate(infected_counts_db)}
    for node in nodes:
        if node not in infected_counts:
            infected_counts[node] = 0

    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE immunity=1 GROUP BY node')
    recovered_counts_db = cursor.fetchall()
    recovered_counts = {values[0]: values[1] for idx, values in enumerate(recovered_counts_db)}
    for node in nodes:
        if node not in recovered_counts:
            recovered_counts[node] = 0

    # print( "Stop report." ) # helps with visually sensing how long this takes.
    return infected_counts, susceptible_counts, recovered_counts 

def update_ages( cursor ):
    cursor.execute('''
        UPDATE agents SET age = age+1/365
    ''')
    return cursor # for pattern

def progress_infections( cursor ):
    # Update infected agents/Progress Existing Infections
    # infection timer: decrement for each infected person
    # Clear Recovereds
    # infected=0, immunity=1, immunity_timer=30-ish
    cursor.execute( "UPDATE agents SET infection_timer = (infection_timer-1) WHERE infection_timer>=1" )
    cursor.execute( "UPDATE agents SET incubation_timer = (incubation_timer-1) WHERE incubation_timer>=1" )
    cursor.execute( "UPDATE agents SET infected=0, immunity=1, immunity_timer=CAST( 10+30*(RANDOM() + 9223372036854775808)/18446744073709551616 AS INTEGER) WHERE infected=1 AND infection_timer=0" )
    return cursor # for pattern

# Update immune agents
def progress_immunities( cursor ):
    # immunity timer: decrement for each immune person
    # immunity flag: clear for each new sus person
    cursor.execute("UPDATE agents SET immunity_timer = (immunity_timer-1) WHERE immunity=1 AND immunity_timer>0" )
    cursor.execute("UPDATE agents SET immunity = 0 WHERE immunity = 1 AND immunity_timer=0" )
    return cursor # for pattern

def calculate_new_infections( cursor, inf, sus ):
    import numpy as np
    node_counts_incubators = np.zeros( settings.num_nodes )
    results = cursor.execute('SELECT node, COUNT(*) FROM agents WHERE incubation_timer >= 1 GROUP BY node').fetchall()
    node_counts_incubators2 = {node: count for node, count in results}
    for node in node_counts_incubators:
        if int(node) in node_counts_incubators2:
            node_counts_incubators += node_counts_incubators2[int(node)]

    sorted_items = sorted(inf.items())
    inf_np = np.array([value for _, value in sorted_items])
    foi = (inf_np-node_counts_incubators) * settings.base_infectivity
    foi = np.array([ min(1,x) for x in foi ])
    sus_np = np.array(list(sus.values()))
    new_infections = list((foi * sus_np).astype(int))
    #print( f"{new_infections} new infections based on foi of\n{foi} and susceptible count of\n{currently_sus}" )
    #print( "new infections = " + str(new_infections) )
    return new_infections 

def handle_transmission( cursor, new_infections, node=0 ):
    # Step 5: Update the infected flag for NEW infectees
    def infect( new_infections, node ):
        #print( f"infect: ni = {new_infections}, node = {node}" )
        cursor.execute("""
            UPDATE agents
            SET infected = 1
            WHERE id in (
                    SELECT id
                    FROM agents
                    WHERE infected=0 AND NOT immunity AND node = :node
                    ORDER BY RANDOM()
                    LIMIT :new_infections
                )""", {'new_infections': int(new_infections), 'node': node } )
    if new_infections>0:
        infect(new_infections, node )
    return cursor # for pattern

    #print( f"{new_infections} new infections in node {node}." )
#with concurrent.futures.ThreadPoolExecutor() as executor:
    #results = list(executor.map(handle_transmission, settings.nodes))

def add_new_infections( cursor ):
    cursor.execute( "UPDATE agents SET infection_timer=CAST( 4+10*(RANDOM() + 9223372036854775808)/18446744073709551616 AS INTEGER) WHERE infected=1 AND infection_timer=0" )
    return cursor # for pattern

def migrate( cursor, timestep, **kwargs ): # ignore kwargs
    # 1% weekly migration ought to cause infecteds from seed node to move to next node
    if timestep % 7 == 0: # every week (or day, depending on what I've set it to)
        cursor.execute( '''
            UPDATE agents SET node = CASE
                WHEN node-1 < 0 THEN :max_node
                ELSE node - 1
            END
            WHERE id IN (
                SELECT id
                    FROM agents
                    WHERE infected=1 AND RANDOM()
                    LIMIT (SELECT COUNT(*) FROM agents) / CAST(1/0.001 AS INTEGER)
                )
            ''', { 'max_node': settings.num_nodes-1 } )
    return cursor # for pattern

# Function to run the simulation for a given number of timesteps
def run_simulation(cursor, csvwriter, num_timesteps):
    #import timeit
    currently_infectious, currently_sus, cur_reco = collect_report( cursor )
    write_timestep_report( csvwriter, 0, currently_infectious, currently_sus, cur_reco )

    for timestep in range(1, num_timesteps + 1):
        update_ages( cursor )

        progress_infections( cursor )

        progress_immunities( cursor )

        new_infections = calculate_new_infections( cursor, currently_infectious, currently_sus )

        for node in nodes:
            handle_transmission( cursor, new_infections[node], node )

        add_new_infections(cursor)
        migrate(cursor, timestep)
        #conn.commit() # using global conn here, a bit of a cheat
        currently_infectious, currently_sus, cur_reco = collect_report( cursor )
        write_timestep_report( csvwriter, 0, currently_infectious, currently_sus, cur_reco )

    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    # Initialize the database
    cursor = initialize_database( conn )
    # Create a CSV file for reporting
    csvfile = open( settings.report_filename, 'w', newline='') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'Recovered'])


    # Run the simulation for 1000 timesteps
    run_simulation( cursor, csvwriter, num_timesteps=duration )

