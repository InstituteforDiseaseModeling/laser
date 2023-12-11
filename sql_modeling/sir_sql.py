import sqlite3
import random
import csv

conn = sqlite3.connect(":memory:")  # Use in-memory database for simplicity

# Function to initialize the SQLite database
def initialize_database():
    cursor = conn.cursor()

    # Create agents table
    cursor.execute('''
        CREATE TABLE agents (
            id INTEGER PRIMARY KEY,
            age INTEGER,
            infected BOOLEAN,
            infection_timer INTEGER,
            incubation_timer INTEGER,
            immunity BOOLEAN,
            immunity_timer INTEGER
        )
    ''')

    # Insert 10,000 agents with random age and all initially uninfected
    agents_data = [(i, random.randint(0, 100), False, 0, 0, False, 0) for i in range(1, 1000000)]
    cursor.executemany('INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?)', agents_data)

    # Seed exactly 100 people to be infected in the first timestep
    cursor.execute('UPDATE agents SET infected = 1, infection_timer=FLOOR(4+10*(RANDOM() + 9223372036854775808)/18446744073709551616), incubation_timer=3 WHERE id IN (SELECT id FROM agents ORDER BY RANDOM() LIMIT 100)')

    conn.commit()
    return conn

def report( timestep, csvwriter ):
    cursor = conn.cursor()
    # Count agents in each state
    cursor.execute('SELECT COUNT(*) FROM agents WHERE infected = 0 AND NOT immunity')
    susceptible_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM agents WHERE infected = 1')
    infected_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM agents WHERE immunity = 1')
    recovered_count = cursor.fetchone()[0]

    # Write the counts to the CSV file
    print( timestep, susceptible_count, infected_count, recovered_count )
    csvwriter.writerow([timestep, susceptible_count, infected_count, recovered_count])
    return infected_count, susceptible_count 

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
            cursor.execute( "UPDATE agents SET infected=0, immunity=1, immunity_timer=FLOOR(10+30*(RANDOM() + 9223372036854775808)/18446744073709551616) WHERE infected=1 AND infection_timer=0" )
        progress_infections()

        # Update immune agents
        def progress_immunities():
            # immunity timer: decrement for each immune person
            # immunity flag: clear for each new sus person
            cursor.execute("UPDATE agents SET immunity_timer = (immunity_timer-1) WHERE immunity=1 AND immunity_timer>0" )
            cursor.execute("UPDATE agents SET immunity = 0 WHERE immunity = 1 AND immunity_timer=0" )
        progress_immunities()

        def handle_transmission():
            # Step 1: Calculate the number of currently infected
            incubating = cursor.execute('SELECT COUNT(*) FROM agents WHERE incubation_timer >= 1').fetchone()[0]

            # Step 2: Calculate the force of infection (foi)
            if( incubating > currently_infectious ):
                raise ValueError( f"incubating = {incubating} found to be > currently_infectious = {currently_infectious}!" )
            #foi = 0.00001 * (currently_infectious-incubating)  # Adjust the multiplier as needed
            foi = 0.0000009 * (currently_infectious-incubating)  # Adjust the multiplier as needed

            # Step 4: Calculate the number of new infections (NEW)
            new_infections = int(foi * currently_sus)
            #print( f"{new_infections} new infections based on foi of {foi} and susceptible cout of {currently_sus}" )

            # Step 5: Update the infected flag for NEW susceptibles
            cursor.execute('''
                UPDATE agents
                SET infected = 1
                WHERE id IN (SELECT id FROM agents WHERE NOT infected AND NOT immunity ORDER BY RANDOM() LIMIT :new_infections)
            ''', {'new_infections': new_infections})

            # handle new infectees, set new infection timer
            cursor.execute( "UPDATE agents SET infection_timer=FLOOR(4+10*(RANDOM() + 9223372036854775808)/18446744073709551616) WHERE infected=1 AND infection_timer=0" )

        handle_transmission()

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
    csvwriter.writerow(['Timestep', 'Susceptible', 'Infected', 'Recovered'])


    # Run the simulation for 1000 timesteps
    run_simulation(connection, csvwriter, num_timesteps=1000 )

