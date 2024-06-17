import sqlite3 as sql
import random
import csv
import concurrent.futures
import numpy as np # not for modeling
from scipy.stats import beta
import pdb
import sys
import os


# We'll fix this settings stuff up soon.
from settings import * # local file
import settings
import demographics_settings 
pop = demographics_settings.pop # slightly tacky way of making this 'globally' available in the module
#print( f"Creating input files for population size {pop}." )

import report
from model_sql import eula

scaled_samples = None

# Globals! (not really)
conn = sql.connect(":memory:")  # Use in-memory database for simplicity
#conn = sql.connect("sir.db")  # Use in-memory database for simplicity
cursor = conn.cursor() # db-specific
# use cursor as model data context; cf dataframe for polars/pandas
#conn = sqlite3.connect("simulation.db")  # Great for inspecting; presumably a bit slower

def get_beta_samples(number):
    # Define parameters
    lifespan_mean = 75
    lifespan_max_value = 110

    # Scale and shift parameters to fit beta distribution
    alpha = 4  # Adjust this parameter to increase lower spread
    beta_ = 2
    samples = beta.rvs(alpha, beta_, size=number)
    scaled_samples = samples * (lifespan_max_value - 1) + 1
    return scaled_samples 

lifespan_idx = 0

def get_node_ids():
    import numpy as np

    array = []
    for node in demographics_settings.nodes:
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

def get_rand_lifespan():
    def beta_lifespan():
        # Generate random samples from the beta distribution
        global scaled_samples 
        if scaled_samples is None:
            scaled_samples = get_beta_samples( pop )

        # Scale samples to match the desired range
        #scaled_samples = samples * max_value
        global lifespan_idx 
        ret_value = scaled_samples[lifespan_idx]
        lifespan_idx += 1
        return ret_value 

    def gaussian():
        mean_lifespan = 75  # Mean lifespan in years
        stddev_lifespan = 10  # Standard deviation of lifespan in years

        # Draw a random number from the normal distribution
        random_lifespan = round(max(np.random.normal(mean_lifespan, stddev_lifespan),0))

    return beta_lifespan()

def init_db_from_csv():
    print( f"Initializing pop from file: {demographics_settings.pop_file}." )
    import pandas as pd
    df = pd.read_csv( demographics_settings.pop_file )
    conn=sql.connect(":memory:")
    df.to_sql("agents",conn,index=False)
    cursor = conn.cursor()
    demographics_settings.nodes = [ x[0] for x in cursor.execute( "SELECT DISTINCT node FROM agents ORDER BY node" ).fetchall() ]
    demographics_settings.num_nodes = len(demographics_settings.nodes)
    demographics_settings.pop = cursor.execute( "SELECT COUNT(*) FROM agents" ).fetchall()[0][0]
    print( f"Loaded population file with {demographics_settings.pop} agents across {demographics_settings.num_nodes} nodes." )
    return cursor

def eula_init( cursor, age_threshold_yrs = 5, eula_strategy="from_db" ):
    print( f"Everyone over the age of {age_threshold_yrs} is permanently immune." )
    # Make everyone over some age perman-immune
    if eula_strategy=="discard":
        print( "TBD: Discard EULAs." )
        cursor.execute( f"DELETE FROM agents WHERE infected=0 AND age > {age_threshold_yrs}" )
        print( f"Leaving {cursor.execute( 'SELECT COUNT(*) FROM agents' ).fetchall()[0][0]} agents." )
    elif eula_strategy=="downsample":
        print( "TBD: Downsample EULAs Not Implemented yet." )
        # Count up all the individuals > age_threshold_yrs by node and age.
        from collections import defaultdict
        results_dict = defaultdict(lambda: defaultdict(int))
        cursor.execute(
                f"""
                SELECT node, CAST(age AS INTEGER) age_int, count(*)
                    FROM agents
                    WHERE age_int > {age_threshold_yrs}
                    GROUP BY node, age_int
                """
            )
        # Sequence here is important
        result = cursor.fetchall()
        cursor.execute( f"DELETE FROM agents WHERE age>{age_threshold_yrs}" )
        for row in result:
            node, age, count = row
            results_dict[node][age] = count
            agents_data = [(node, age, False, 0, 0, True, -1, 999999, count)]
            cursor.executemany('INSERT INTO agents VALUES (null, ?, ?, ?, ?, ?, ?, ?, ?, ?)', agents_data)
    elif eula_strategy=="separate":
        # Create a new table eula with the same schema as the agents table
        cursor.execute( "CREATE TABLE eula AS SELECT * FROM agents WHERE 1=0" )
        # Move rows where age > some_age_threshold into the eula table
        cursor.execute( f"INSERT INTO eula SELECT * FROM agents WHERE age > {age_threshold_yrs}" )
        cursor.execute( f"CREATE INDEX idx_node ON eula (node)" )
        # Delete rows from the agents table where age > some_age_threshold
        cursor.execute( f"DELETE FROM agents WHERE age > {age_threshold_yrs}" )
    elif eula_strategy=="from_file":
        print( "EULAed agents were pre-sorted into separate file which we are loading into a table now" )
        import pandas as pd
        df = pd.read_csv( demographics_settings.eulad_pop_file )
        conn=sql.connect(":memory:")
        df.to_sql("agents",conn,index=False)
        cursor = conn.cursor()
        demographics_settings.pop = cursor.execute( "SELECT COUNT(*) FROM agents" ).fetchall()[0][0]
        print( f"Loaded population file with {demographics_settings.pop} agents across {demographics_settings.num_nodes} nodes." )
    elif eula_strategy=="from_db":
        eula.init()
    elif not eula_strategy:
        print( "TBD: Keeping EULAs." )
        cursor.execute( f"UPDATE agents SET immunity = 1, immunity_timer=-1 WHERE infected=0 AND age > {age_threshold_yrs}" )
    else:
        raise ValueError( f"Unknown eula strategy: {eula_strategy}." )
    return cursor

# Function to initialize the SQLite database
def initialize_database( conn=None, from_file=True ):
    # TBD: Make programmatic option to init db instead of load from csv.
    if from_file:
        return init_db_from_csv()

    print( "Initializing pop NOT from file." )

    if not conn:
        conn = sql.connect(":memory:")  # Use in-memory database for simplicity
    cursor = conn.cursor()

    # Create agents table
    cursor.execute('''
        CREATE TABLE agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node INTEGER,
            age REAL,
            infected BOOLEAN,
            infection_timer INTEGER,
            incubation_timer INTEGER,
            immunity BOOLEAN,
            immunity_timer INTEGER,
            expected_lifespan INTEGER
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

    agents_data = [(node_assignments[i], random.randint(0, 100)+random.randint(0,365)/365.0, False, 0, 0, False, 0, get_rand_lifespan()) for i in range(pop)]
    cursor.executemany('INSERT INTO agents VALUES (null, ?, ?, ?, ?, ?, ?, ?, ?)', agents_data)

    # Seed exactly 100 people to be infected in the first timestep
    # uniform distribution draws seem a bit clunky in SQLite. Just looking for values from 4 to 14. Ish.
    cursor.execute( 'UPDATE agents SET infected = 1, infection_timer=9+RANDOM()%6, incubation_timer=3 WHERE id IN (SELECT id FROM agents WHERE node=:big_node ORDER BY RANDOM() LIMIT 100)', { 'big_node': demographics_settings.num_nodes-1 } )
    #for node in range( demographics_settings.num_nodes ):
        #cursor.execute( 'UPDATE agents SET infected = 1, infection_timer=9+RANDOM()%6, incubation_timer=3 WHERE id IN (SELECT id FROM agents WHERE node=:big_node ORDER BY RANDOM() LIMIT 100)', { 'big_node': node } )

    conn.commit()

    return cursor

def collect_report( cursor ):
    #print( "Start report." ) # helps with visually sensing how long this takes.
    # Count agents in each state
    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE infected=0 AND immunity=0 GROUP BY node')
    # TBD: Find better way to get node list
    
    susceptible_counts_db = cursor.fetchall()
    susceptible_counts = {values[0]: values[1] for idx, values in enumerate(susceptible_counts_db)}
    for node in demographics_settings.nodes:
        if node not in susceptible_counts:
            susceptible_counts[node] = 0

    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE infected=1 GROUP BY node')
    infected_counts_db = cursor.fetchall()
    infected_counts = {values[0]: values[1] for idx, values in enumerate(infected_counts_db)}
    for node in demographics_settings.nodes:
        if node not in infected_counts:
            infected_counts[node] = 0

    cursor.execute('SELECT node, COUNT(*) FROM agents WHERE immunity=1 GROUP BY node')
    recovered_counts_db = cursor.fetchall()
    recovered_counts = {values[0]: values[1] for idx, values in enumerate(recovered_counts_db)}
    for node in demographics_settings.nodes:
        if node not in recovered_counts:
            recovered_counts[node] = 0

    recovered_counts_db = eula.get_recovereds_by_node()
    for key, count in recovered_counts_db.items():
        recovered_counts[key] += count
    # print( "Stop report." ) # helps with visually sensing how long this takes.
    return infected_counts, susceptible_counts, recovered_counts 

def update_ages( cursor, totals=None ): # totals are for demographic-based fertility
    global eula_cursor
    #cursor.execute("UPDATE agents SET age = age+1/365.0")
    #eula_cursor.execute("UPDATE agents SET age = age+1/365.0")
    def births():
        def add_newborns( node, babies ):
            agents_data = [(node, 0, False, 0, 0, False, 0, get_rand_lifespan()) for i in range(babies)]
            cursor.executemany('INSERT INTO agents VALUES (null, ?, ?, ?, ?, ?, ?, ?, ?)', agents_data)

        # I want to move this code into seperate dedicate mini submodule
        def demographic_dependent():
            # Births: Let's aim for 100 births per 1,000 woman of cba per year. So 
            # 0.1/365 per day or 2.7e-4
            wocba = cursor.execute( "SELECT node, COUNT(*)/2 FROM agents WHERE age>15 and age<45 GROUP BY node ORDER BY node").fetchall()
            wocba  = {values[0]: values[1] for idx, values in enumerate(wocba)}
            for node,count in wocba.items():
                newbabies = np.sum( np.random.rand(count) <  2.7e-4)
                #print( f"node,newborns = {node},{newbabies}" )
                if newbabies > 0:
                    add_newborns( node, newbabies )

        def births_from_cbr( node_pops, rate=30 ):
            # TBD: births = CBR & node_pop / 1000
            # placeholder: just say 10 per node for now to test rest of code path
            new_babies = {}
            for node in node_pops:
                cbr_node = rate * (node_pops[node]/1000.0)/365.0
                new_babies[node] = np.random.poisson( cbr_node )
            return new_babies

        #print( f"pop after births = {num_agents}" )
        new_babies = births_from_cbr( totals, rate=demographics_settings.cbr )
        #print( f"New babies by node: {new_babies}" )
        # Iterate over nodes and add newborns
        for node, count in new_babies.items():
            if count > 0:
                add_newborns(node, count)

    def deaths():
        # Deaths
        #cursor.execute( "UPDATE agents SET infected=0, immunity=1, immunity_timer=-1 WHERE age>expected_lifespan" )
        #eula_cursor.execute( "DELETE FROM agents WHERE age>=expected_lifespan" )
        #num_agents = cursor.execute( "SELECT COUNT(*) FROM agents" ).fetchall()[0][0] 
        # NOTE: Only doing R->D for now
        new_deaths = eula.update_natural_mortality()
        #print( f"pop after deaths = {num_agents}" )
    births()
    deaths()
    return cursor # for pattern

def progress_infections( cursor ):
    # Update infected agents/Progress Existing Infections
    # infection timer: decrement for each infected person
    # Clear Recovereds
    # infected=0, immunity=1, immunity_timer=30-ish
    cursor.execute( "UPDATE agents SET infection_timer = (infection_timer-1) WHERE infection_timer>=1" )
    cursor.execute( "UPDATE agents SET incubation_timer = (incubation_timer-1) WHERE incubation_timer>=1" )
    cursor.execute( "UPDATE agents SET infected=0, immunity=1, immunity_timer=(20+RANDOM()%10) WHERE infected=1 AND infection_timer<=0" )
    #cursor.execute( "UPDATE agents SET infected=0, immunity=1, immunity_timer=(11+RANDOM()%10) WHERE infected=1 AND infection_timer=0" )
    return cursor # for pattern

# Update immune agents
def progress_immunities( cursor ):
    # immunity timer: decrement for each immune person
    # immunity flag: clear for each new sus person
    cursor.execute("UPDATE agents SET immunity_timer = (immunity_timer-1) WHERE immunity=1 AND immunity_timer>0" )
    cursor.execute("UPDATE agents SET immunity = 0 WHERE immunity = 1 AND immunity_timer<=0" )
    return cursor # for pattern

def calculate_new_infections( cursor, inf, sus, totals ):
    import numpy as np
    node_counts_incubators = np.zeros( demographics_settings.num_nodes )
    results = cursor.execute('SELECT node, COUNT(*) FROM agents WHERE incubation_timer >= 1 GROUP BY node').fetchall()
    node_counts_incubators2 = {node: count for node, count in results}

    # ni is an array; inf and sus are dicts
    new_infections = np.ones(len(inf))*settings.base_infectivity
    # Maybe doesn't need to be separate?
    #pdb.set_trace()
    for node in range(len(new_infections)):
        exposed_fraction = 0
        if node in node_counts_incubators2:
            exposed_fraction = node_counts_incubators2[node]/totals[node]
            #print( f"Node {node} has {node_counts_incubators2[node]} incubators." )
        inf[node] -= exposed_fraction 
        new_infections[node] = round(inf[node]*sus[node]*new_infections[node])

    #print( f"{new_infections} new infections based on inf of\n{[inf[x] for x in sorted(inf)]} and susceptible fraction of\n{[sus[x] for x in sorted(sus)]}" )
    return new_infections

def _handle_transmission_inner( cursor, new_infections, node=0 ):
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
    #results = list(executor.map(handle_transmission, demographics_settings.nodes))

def handle_transmission( df, new_infections ):
    for node in demographics_settings.nodes:
        if new_infections[ node ] > 0:
            df = _handle_transmission_inner( df, new_infections[ node ], node )
    return df

def add_new_infections( cursor ):
    # 11-30 is what numpy is doing
    cursor.execute( "UPDATE agents SET incubation_timer=3, infection_timer=8+RANDOM()%5 WHERE infected=1 AND infection_timer=0" )
    #cursor.execute( "UPDATE agents SET incubation_timer=11, infection_timer=20+RANDOM()%10 WHERE infected=1 AND infection_timer=0" )
    #print( f"Set infection timer for {cursor.rowcount} agents." )
    return cursor # for pattern

def distribute_interventions( cursor, timestep ):
    # Give vaccine to individuals turning 9 months in node 15. Should cut off transmission.
    def ria_9mo():
        query = """
            UPDATE agents
            SET immunity = 1,
                immunity_timer = 3650
            WHERE
                age > 290/365.0
                AND age < 291/365.0
                AND immunity = 0
                AND node = 15
            """
        cursor.execute( query )
        if cursor.rowcount > 0:
            print( f"DEBUG: Distributed {cursor.rowcount} 20-year acquisition-blocking vaccines to 9 month olds." )
    def campaign( coverage = 1.0 ):
        query = f"""
            UPDATE agents
            SET immunity = 1,
                immunity_timer = -1
            WHERE
                immunity = 0
                AND age < 16
                AND node = 15
            ORDER BY RANDOM()
            LIMIT CAST((SELECT COUNT(*) FROM agents WHERE ( 
                immunity = 0
                AND age < 16
                AND node = 15
            ) ) * {coverage} AS INTEGER)
        """
        cursor.execute( query )
        print( f"Vaccinated {cursor.rowcount} in node 15 at timestep 15." )
    #ria_9mo()
    if timestep == settings.campaign_day:
        campaign()
    return cursor

def migrate( cursor, timestep, **kwargs ): # ignore kwargs
    # 1% weekly migration ought to cause infecteds from seed node to move to next node
    if timestep % settings.migration_interval == 0: # every week (or day, depending on what I've set it to)
        cursor.execute( '''
            UPDATE agents SET node = CASE
                WHEN node = 0 THEN :max_node
                ELSE node - 1
            END
            WHERE id IN (
                SELECT id
                    FROM agents
                    WHERE infected=1 
                    ORDER BY RANDOM()
                    LIMIT CAST( (SELECT COUNT(*) FROM agents WHERE infected=1 ) * 0.05 AS INTEGER)
                )
            ''', { 'max_node': demographics_settings.num_nodes-1 } )
    return cursor # for pattern

# Function to run the simulation for a given number of timesteps
def run_simulation(cursor, csvwriter, num_timesteps):
    #import timeit
    currently_infectious, currently_sus, cur_reco = collect_report( cursor )
    report.write_timestep_report( csvwriter, 0, currently_infectious, currently_sus, cur_reco )

    for timestep in range(1, num_timesteps + 1):
        update_ages( cursor )

        progress_infections( cursor )

        progress_immunities( cursor )

        new_infections = calculate_new_infections( cursor, currently_infectious, currently_sus )

        #for node in nodes:
            #handle_transmission( cursor, new_infections[node], node )
        handle_transmission( cursor, new_infections[node] )

        add_new_infections(cursor)
        migrate(cursor, timestep)
        #conn.commit() # using global conn here, a bit of a cheat
        currently_infectious, currently_sus, cur_reco = collect_report( cursor )
        report.write_timestep_report( csvwriter, timestep, currently_infectious, currently_sus, cur_reco )

    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    # Initialize the database
    cursor = initialize_database( conn, from_file=False )
    #cursor = init_db_from_csv( conn )
    # Create a CSV file for reporting
    csvfile = open( settings.report_filename, 'w', newline='') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'Recovered'])


    # Run the simulation for 1000 timesteps
    run_simulation( cursor, csvwriter, num_timesteps=duration )

