import sqlite3 as sql
import random
import csv
import concurrent.futures
import numpy as np # not for modeling
from scipy.stats import beta
import json
import pdb
import sys
import os
import importlib.resources as pkg_resources
import idmlaser # seems odd and circular


# We'll fix this settings stuff up soon.
from settings import * # local file
import settings
import demographics_settings 
pop = demographics_settings.pop # slightly tacky way of making this 'globally' available in the module
#print( f"Creating input files for population size {pop}." )

from . import report
from .model_sql import eula

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


# Function to map JSON schema types to SQLite types
def map_json_type_to_sqlite(json_type):
    if json_type == 'integer':
        return 'INTEGER'
    elif json_type == 'number':
        return 'REAL'
    elif json_type == 'boolean':
        return 'BOOLEAN'
    else:
        raise ValueError(f"Unsupported JSON type: {json_type}")

# Construct the CREATE TABLE SQL statement
def construct_create_table_sql(schema):
    table_name = schema['title']
    columns = schema['properties']
    required = schema.get('required', [])

    column_defs = []
    for column_name, column_info in columns.items():
        column_type = map_json_type_to_sqlite(column_info['type'])
        if column_name == 'id':
            column_def = f"{column_name} {column_type} PRIMARY KEY AUTOINCREMENT"
        else:
            column_def = f"{column_name} {column_type}"
            if column_name in required:
                column_def += " NOT NULL"
        column_defs.append(column_def)

    column_defs_str = ",\n    ".join(column_defs)
    create_table_sql = f"CREATE TABLE {table_name} (\n    {column_defs_str}\n);"

    return create_table_sql


# Function to initialize the SQLite database
def initialize_database( conn=None, from_file=True ):
    # TBD: Make programmatic option to init db instead of load from csv.
    if from_file:
        return init_db_from_csv()

    print( "Initializing pop NOT from file." )

    if not conn:
        conn = sql.connect(":memory:")  # Use in-memory database for simplicity
    cursor = conn.cursor()

    # Load schema.json
    #with open('schema.json', 'r') as file:
    with pkg_resources.open_text(idmlaser, 'schema.json') as file:
        schema = json.load(file)
    
    # Create the table
    create_table_sql = construct_create_table_sql(schema)
    #print(create_table_sql)  # Print the SQL statement for debugging purposes
    with conn:
        conn.execute(create_table_sql)

    #print("Table created successfully.")

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

