import sqlite3
import random
import csv

# Globals! (not really)
#conn = sqlite3.connect(":memory:")  # Use in-memory database for simplicity
conn = sqlite3.connect("simulation.db")  # Use in-memory database for simplicity
#pop = 1000000
pop = int(1e6)
num_nodes = 20
nodes = [ x for x in range(num_nodes) ]
duration = 200 # 1000
import pdb

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
    print(array[:20])

    return array

# Function to initialize the SQLite database
def initialize_database():
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
    # Insert 10,000 agents with random age and all initially uninfected
    #agents_data = [(i, random.randint(0, num_nodes-1), random.randint(0, 100), False, 0, 0, False, 0) for i in range(1, pop)]
    node_assignments = get_node_ids()
    agents_data = [(i, node_assignments[i], random.randint(0, 100), False, 0, 0, False, 0) for i in range(1, pop)]
    cursor.executemany('INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?)', agents_data)

    # Seed exactly 100 people to be infected in the first timestep
    cursor.execute('UPDATE agents SET infected = 1, infection_timer=FLOOR(4+10*(RANDOM() + 9223372036854775808)/18446744073709551616), incubation_timer=3 WHERE id IN (SELECT id FROM agents ORDER BY RANDOM() LIMIT 100)')

    conn.commit()

    return conn

def report( timestep, csvwriter ):
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
    for node in nodes:
        csvwriter.writerow([timestep,
            node,
            susceptible_counts[node] if node in susceptible_counts else 0,
            infected_counts[node] if node in infected_counts else 0,
            recovered_counts[node] if node in recovered_counts else 0,
            ]
        )
    return infected_counts, susceptible_counts

# Function to run the simulation for a given number of timesteps
def run_simulation(conn, csvwriter, num_timesteps):
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

        def progress_infections():
            # Progress Existing Infections
            # Clear Recovereds
            # infected=0, immunity=1, immunity_timer=30-ish
            cursor.execute( "UPDATE agents SET infection_timer = (infection_timer-1) WHERE infection_timer>=1" )
            cursor.execute( "UPDATE agents SET incubation_timer = (incubation_timer-1) WHERE incubation_timer>=1" )
            cursor.execute( "UPDATE agents SET infected=False, immunity=1, immunity_timer=FLOOR(10+30*(RANDOM() + 9223372036854775808)/18446744073709551616) WHERE infected=True AND infection_timer=0" )
        progress_infections()

        # Update immune agents
        def progress_immunities():
            # immunity timer: decrement for each immune person
            # immunity flag: clear for each new sus person
            cursor.execute("UPDATE agents SET immunity_timer = (immunity_timer-1) WHERE immunity=1 AND immunity_timer>0" )
            cursor.execute("UPDATE agents SET immunity = 0 WHERE immunity = 1 AND immunity_timer=0" )
        progress_immunities()

        def handle_transmission( node=0 ):
            # Step 1: Calculate the number of currently infected
            incubating = cursor.execute('SELECT COUNT(*) FROM agents WHERE incubation_timer >= 1 and node=:node', { 'node': node }).fetchone()[0]

            # Step 2: Calculate the force of infection (foi)
            if node in currently_infectious:
                if( incubating > currently_infectious[node] ):
                    raise ValueError( f"incubating = {incubating} found to be > currently_infectious = {currently_infectious[node]} in node {node}!" )
                foi = 0.00001 * (currently_infectious[node]-incubating)  # Adjust the multiplier as needed
            else:
                foi = 0
            #foi = 0.0000009 * (currently_infectious[node]-incubating)  # Adjust the multiplier as needed
            #foi = 0.9 * pop * (currently_infectious[node]-incubating)  # Adjust the multiplier as needed

            # Step 4: Calculate the number of new infections (NEW)
            if node in currently_sus:
                new_infections = int(foi * currently_sus[node])
            else:
                new_infections = 0

            #print( f"{new_infections} new infections based on foi of {foi} and susceptible cout of {currently_sus}" )

            # Step 5: Update the infected flag for NEW infectees
            cursor.execute('''
                UPDATE agents
                SET infected = True
                WHERE id IN (SELECT id FROM agents WHERE NOT infected AND NOT immunity AND node=:node ORDER BY RANDOM() LIMIT :new_infections)
            ''', {'new_infections': new_infections, 'node': node })
            #print( f"{new_infections} new infections in node {node}." )

        for node in nodes:
            handle_transmission( node )
            # handle new infectees, set new infection timer
            cursor.execute( "UPDATE agents SET infection_timer=FLOOR(4+10*(RANDOM() + 9223372036854775808)/18446744073709551616) WHERE infected AND infection_timer=0" )

        conn.commit()
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

