import sqlite3
import random
import csv
import concurrent.futures
import pdb
import sys

from settings import * # local file
import settings

settings.base_infectivity = 0.00001
write_report = True # sometimes we want to turn this off to check for non-reporting bottlenecks

# Globals! (not really)
#conn = sqlite3.connect(":memory:")  # Use in-memory database for simplicity
conn = sqlite3.connect("simulation.db")  # Use in-memory database for simplicity
def get_node_ids():
    import numpy as np

    array = []
    for node in nodes:
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
def initialize_database():
    print( "Initializing pop NOT from file." )
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
    #cursor.execute( "CREATE INDEX idx_agents_node ON agents(id, node)" )
    #cursor.execute( "CREATE INDEX idx_agents_node_infected ON agents(node, infected)" )
    #cursor.execute( "CREATE INDEX idx_agents_node_immunity ON agents(node, immunity)" )
                

    # Insert 10,000 agents with random age and all initially uninfected
    #agents_data = [(i, random.randint(0, num_nodes-1), random.randint(0, 100), False, 0, 0, False, 0) for i in range(1, pop)]
    node_assignments = get_node_ids()
    agents_data = [(i, node_assignments[i], random.randint(0, 100), False, 0, 0, False, 0) for i in range(1, pop)]
    cursor.executemany('INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?)', agents_data)

    # Seed exactly 100 people to be infected in the first timestep
    cursor.execute( 'UPDATE agents SET infected = 1, infection_timer=FLOOR(4+10*(RANDOM() + 9223372036854775808)/18446744073709551616), incubation_timer=3 WHERE id IN (SELECT id FROM agents WHERE node=:big_node ORDER BY RANDOM() LIMIT 100)', { 'big_node': num_nodes-1 } )

    conn.commit()

    return conn

def report( timestep, csvwriter ):
    print( "Start report." )
    cursor = conn.cursor()
    # Count agents in each state
    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE NOT infected AND NOT immunity GROUP BY node')
    # this seems slow and clunky
    
    susceptible_counts_db = cursor.fetchall()
    susceptible_counts = {values[0]: values[1] for idx, values in enumerate(susceptible_counts_db)}
    for node in nodes:
        if node not in susceptible_counts:
            susceptible_counts[node] = 0

    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE infected GROUP BY node')
    infected_counts_db = cursor.fetchall()
    infected_counts = {values[0]: values[1] for idx, values in enumerate(infected_counts_db)}
    for node in nodes:
        if node not in infected_counts:
            infected_counts[node] = 0

    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE immunity GROUP BY node')
    recovered_counts_db = cursor.fetchall()
    recovered_counts = {values[0]: values[1] for idx, values in enumerate(recovered_counts_db)}
    for node in nodes:
        if node not in recovered_counts:
            recovered_counts[node] = 0

    # Write the counts to the CSV file
    print( f"T={timestep}, S={susceptible_counts}, I={infected_counts}, R={recovered_counts}" )
    if write_report:
        for node in nodes:
            csvwriter.writerow([timestep,
                node,
                susceptible_counts[node] if node in susceptible_counts else 0,
                infected_counts[node] if node in infected_counts else 0,
                recovered_counts[node] if node in recovered_counts else 0,
                ]
            )
    print( "Stop report." )
    return infected_counts, susceptible_counts

# Function to run the simulation for a given number of timesteps
def run_simulation(conn, csvwriter, num_timesteps):
    import timeit
    currently_infectious, currently_sus = report( 0, csvwriter )
    cursor = conn.cursor()

    for timestep in range(1, num_timesteps + 1):
        # Update infected agents
        # infection timer: decrement for each infected person

        def update_ages():
            cursor.execute('''
                UPDATE agents SET age = age+1/365
            ''')
        update_ages()
        #print( "Back from...update_ages()" )
        #print( f"update_ages took {age_time}" )

        def progress_infections():
            # Progress Existing Infections
            # Clear Recovereds
            # infected=0, immunity=1, immunity_timer=30-ish
            cursor.execute( "UPDATE agents SET infection_timer = (infection_timer-1) WHERE infection_timer>=1" )
            cursor.execute( "UPDATE agents SET incubation_timer = (incubation_timer-1) WHERE incubation_timer>=1" )
            cursor.execute( "UPDATE agents SET infected=False, immunity=1, immunity_timer=FLOOR(10+30*(RANDOM() + 9223372036854775808)/18446744073709551616) WHERE infected=True AND infection_timer=0" )
        progress_infections()
        #print( "Back from...progress_infections()" )

        # Update immune agents
        def progress_immunities():
            # immunity timer: decrement for each immune person
            # immunity flag: clear for each new sus person
            cursor.execute("UPDATE agents SET immunity_timer = (immunity_timer-1) WHERE immunity=1 AND immunity_timer>0" )
            cursor.execute("UPDATE agents SET immunity = 0 WHERE immunity = 1 AND immunity_timer=0" )
        progress_immunities()
        #print( "Back from...progress_immunities()" )

        import numpy as np
        node_counts_incubators = np.zeros( settings.num_nodes )
        results = cursor.execute('SELECT node, COUNT(*) FROM agents WHERE incubation_timer >= 1 GROUP BY node').fetchall()
        node_counts_incubators2 = {node: count for node, count in results}
        for node in node_counts_incubators:
            if int(node) in node_counts_incubators2:
                node_counts_incubators += node_counts_incubators2[int(node)]

        sorted_items = sorted(currently_infectious.items())
        inf_np = np.array([value for _, value in sorted_items])
        foi = (inf_np-node_counts_incubators) * settings.base_infectivity
        foi = np.array([ min(1,x) for x in foi ])
        sus_np = np.array(list(currently_sus.values()))
        new_infections = list((foi * sus_np).astype(int))
        #print( f"{new_infections} new infections based on foi of\n{foi} and susceptible count of\n{currently_sus}" )
        print( "new infections = " + str(new_infections) )

        def handle_transmission( node=0 ):
            # Step 5: Update the infected flag for NEW infectees
            def infect( new_infections, node ):
                #print( f"infect: ni = {new_infections}, node = {node}" )
                cursor.execute("""
                    UPDATE agents
                    SET infected = True
                    WHERE id in (
                            SELECT id
                            FROM agents
                            WHERE NOT infected AND NOT immunity AND node = :node
                            ORDER BY RANDOM()
                            LIMIT :new_infections
                        )""", {'new_infections': int(new_infections), 'node': node } )
            if new_infections[node]>0:
                infect(new_infections[node], node )

            #print( f"{new_infections} new infections in node {node}." )
        #with concurrent.futures.ThreadPoolExecutor() as executor:
            #results = list(executor.map(handle_transmission, settings.nodes))

        for node in nodes:
            handle_transmission( node )
            #print( "Back from...handle_transmission()" )
            # handle new infectees, set new infection timer
        print( "Back from creating new infections." )
        cursor.execute( "UPDATE agents SET infection_timer=FLOOR(4+10*(RANDOM() + 9223372036854775808)/18446744073709551616) WHERE infected AND infection_timer=0" )
        #print( "Back from...init_inftimers()" )

        def migrate():
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
                            WHERE infected AND RANDOM()
                            LIMIT (SELECT COUNT(*) FROM agents) / CAST(1/0.001 AS INTEGER)
                        )
                    ''', { 'max_node': num_nodes-1 } )
        migrate()
        conn.commit()
        #print( "Back from...commit()" )
        #print( f"{cursor.execute('select * from agents where infected limit 25').fetchall()}".replace("), ",")\n") )
        #print( "*****" )
        currently_infectious, currently_sus = report( timestep, csvwriter )



    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    # Initialize the database
    connection = initialize_database()
    # Create a CSV file for reporting
    csvfile = open('simulation_report.csv', 'w', newline='') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'Recovered'])


    # Run the simulation for 1000 timesteps
    run_simulation(connection, csvwriter, num_timesteps=duration )

