import sqlite3
import random
from tqdm import tqdm
import csv
import pdb

# Simulation parameters
num_timesteps = 3650 # 36500

# Connect to SQLite database (create one if it doesn't exist)
conn = sqlite3.connect(":memory:")

def init_db( conn ):
    cursor = conn.cursor()
    # Create the 'agents' table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY,
            dob INTEGER,
            active BOOLEAN,
            test INTEGER
        )
    ''')

    # Insert 1e7 agents into the table with random dob and initial active status
    num_agents = int(1e7)

    for i in range(num_agents):
        dob = random.randint(-36500, 36500)
        active = ( dob >= -5 * 365 and dob <= 0 )
        test = 0
        cursor.execute('''
            INSERT INTO agents (dob, active, test) VALUES (?, ?, ?)
        ''', (dob, active, test))
    cursor.execute( "CREATE TABLE ordered_agents AS SELECT * FROM agents ORDER BY dob" )
    cursor.execute( "DROP TABLE agents" )
    #cursor.execute( "CREATE INDEX idx_dob ON ordered_agents (dob)" ) # doesn't make much measurable difference after sorting
    cursor.execute( "CREATE INDEX idx_active ON ordered_agents(active)" ) # very important
    cursor.execute( "CREATE INDEX idx_dob_active ON ordered_agents(dob, active)" ) # pretty important

    # Commit changes
    conn.commit()

def write_db_to_csv( conn ):
    cursor = conn.cursor()
    get_all_query = "SELECT * FROM ordered_agents"
    cursor.execute( get_all_query )
    rows = cursor.fetchall()

    print( f"Writing population file out out to csv: mostly_inactive.csv." )
    with open( "mostly_inactive.csv", "w", newline='' ) as csvfile:
        csv_writer = csv.writer( csvfile )
        csv_writer.writerow( ['dob', 'active', 'test' ] )
        csv_writer.writerows( rows )

def run( conn ):
    # Simulation loop
    for timestep in tqdm(range(1, num_timesteps + 1)):
        #actives = cursor.execute( "SELECT COUNT(*) FROM ordered_agents WHERE active=1" ).fetchall()
        #print( actives[0] )

        # Update active status based on dob and current timestep
        cursor.execute(f"UPDATE ordered_agents SET active = true WHERE dob = {-timestep}" )
        print( f"{cursor.rowcount} newborns." )

        # Increment 'test' value for active ordered_agents
        cursor.execute('''
            UPDATE ordered_agents
            SET test = test + 1
            WHERE active = 1
        ''')
        print( f"{cursor.rowcount} actives updated." )

        # Deactivate 0.1% of active ordered_agents randomly
        cursor.execute('''
            UPDATE ordered_agents SET active=false WHERE active=true ORDER BY RANDOM() LIMIT ( SELECT COUNT(*) FROM ordered_agents WHERE active=true )/1000
        ''')
        print( f"{cursor.rowcount} fewer actives." )

        # Commit changes at each timestep
        conn.commit()


if __name__ == "__main__":
    init_db( conn )
    run( conn )
    # Close the database connection
    conn.close()
