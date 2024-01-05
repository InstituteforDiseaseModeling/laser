import sqlite3
import random
from tqdm import tqdm
import pdb

# Connect to SQLite database (create one if it doesn't exist)
#conn = sqlite3.connect('agents.db')
conn = sqlite3.connect(":memory:")
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
initial_active_percentdob = 0.1
for i in range(num_agents):
    dob = random.randint(-36500, 36500)
    active = ( dob <= 5 * 365 and dob > 0 )
    test = 0
    cursor.execute('''
        INSERT INTO agents (dob, active, test) VALUES (?, ?, ?)
    ''', (dob, active, test))

# Commit changes
conn.commit()

print( "DB setup with 1e7 agents." )

# Simulation parameters
num_timesteps = 3650 # 36500

# Simulation loop
for timestep in tqdm(range(1, num_timesteps + 1)):
    # Update active status based on dob and current timestep
    cursor.execute(f"UPDATE agents SET active = 1 WHERE dob = {-timestep}" )

    # Increment 'test' value for active agents
    cursor.execute('''
        UPDATE agents
        SET test = test + 1
        WHERE active = 1
    ''')

    # Deactivate 0.1% of active agents randomly
    cursor.execute('''
        UPDATE agents
        SET active = 0
        WHERE active = 1 AND RANDOM() < ?
    ''', (0.001,))

    actives = cursor.execute( "SELECT COUNT(*) FROM agents WHERE active=1" ).fetchall()
    print( actives )

    # Commit changes at each timestep
    conn.commit()

# Close the database connection
conn.close()

