import random
import csv
import numpy as np
import settings
import concurrent.futures
import pdb

# Globals! (not really)
#pop = 1000000
#pop = int(5e5)
#num_nodes = 200
#nodes = [ x for x in range(num_nodes) ]
#duration = 365 # 1000
#base_infectivity = 0.00001
base_infectivity = 0.000002
safe_mode = False

def load():

    # Replace 'your_file.csv' with the actual path to your CSV file
    #csv_file = 'pop_500k_200nodes.csv'
    csv_file = settings.pop_file

    # Load the entire CSV file into a NumPy array
    header_row = np.genfromtxt(csv_file, delimiter=',', dtype=str, max_rows=1)

    # Load the remaining data as numerical values, skipping the header row
    data = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=1)

    # Extract headers from the header row
    headers = header_row

    # Load each column into a separate NumPy array
    columns = {header: data[:, i] for i, header in enumerate(headers)}
    columns['infected'] = columns['infected'].astype(bool)
    columns['immunity'] = columns['immunity'].astype(bool)
    columns['node'] = columns['node'].astype(int)

    settings.pop = len(columns['infected'])
    print( f"Population={settings.pop}" )
    settings.nodes = [ node for node in np.unique(columns['node']) ]
    settings.num_nodes = len(settings.nodes)
    print( f"Nodes={settings.num_nodes}" )
    # Now 'columns' is a dictionary where keys are column headers and values are NumPy arrays
    return columns

    
def report( data, timestep, csvwriter ):
    print( "Start report." )
    # cursor.execute('SELECT node, COUNT(*) FROM agents WHERE NOT infected AND NOT immunity GROUP BY node')
    # this seems slow and clunky

    condition_mask = np.logical_and(~data['infected'], ~data['immunity'])
    unique_nodes, counts = np.unique(data['node'][condition_mask], return_counts=True)

    # Display the result
    susceptible_counts_db = list(zip(unique_nodes, counts))
    susceptible_counts = {values[0]: values[1] for idx, values in enumerate(susceptible_counts_db)}
    for node in settings.nodes:
        if node not in susceptible_counts:
            susceptible_counts[node] = 0

    unique_nodes, counts = np.unique(data['node'][data['infected']], return_counts=True)
    infected_counts_db = list(zip(unique_nodes, counts))
    #cursor.execute('SELECT node, COUNT(*) FROM agents WHERE infected GROUP BY node')
    infected_counts = {values[0]: values[1] for idx, values in enumerate(infected_counts_db)}
    for node in settings.nodes:
        if node not in infected_counts:
            infected_counts[node] = 0

    #cursor.execute('SELECT node, COUNT(*) FROM agents WHERE immunity GROUP BY node')
    #recovered_counts_db = cursor.fetchall()
    unique_nodes, counts = np.unique(data['node'][data['immunity']], return_counts=True)
    recovered_counts_db  = list(zip(unique_nodes, counts))
    recovered_counts = {values[0]: values[1] for idx, values in enumerate(recovered_counts_db)}
    for node in settings.nodes:
        if node not in recovered_counts:
            recovered_counts[node] = 0

    # Write the counts to the CSV file
    #print( f"T={timestep}, S={susceptible_counts}, I={infected_counts}, R={recovered_counts}" )
    print( f"T={timestep}, I={infected_counts}" )
    for node in settings.nodes:
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
def run_simulation(data, csvwriter, num_timesteps):
    currently_infectious, currently_sus = report( data, 0, csvwriter )


    for timestep in range(1, num_timesteps + 1):
        # Update infected agents
        # infection timer: decrement for each infected person

        def update_ages():
            cursor.execute("UPDATE agents SET age = age+1/365")
        def update_ages_np():
            data['age'] += 1/365
        update_ages_np()
        #print( "Back from...update_ages()" )
        #print( f"update_ages took {age_time}" )

        def progress_infections():
            # Progress Existing Infections
            # Clear Recovereds
            # infected=0, immunity=1, immunity_timer=30-ish
            cursor.execute( "UPDATE agents SET infection_timer = (infection_timer-1) WHERE infection_timer>=1" )
            cursor.execute( "UPDATE agents SET incubation_timer = (incubation_timer-1) WHERE incubation_timer>=1" )
            cursor.execute( "UPDATE agents SET infected=False, immunity=1, immunity_timer=FLOOR(10+30*(RANDOM() + 9223372036854775808)/18446744073709551616) WHERE infected=True AND infection_timer=0" )
        def progress_infections_np():
            data['infection_timer'][data['infection_timer'] >= 1] -= 1
            data['incubation_timer'][data['incubation_timer'] >= 1] -= 1
            condition = np.logical_and(data['infected'], data['infection_timer'] == 0)
            data['infected'][condition] = False
            data['immunity_timer'][condition] = np.random.randint(10, 41, size=np.sum(condition))

        #progress_infections()
        progress_infections_np()
        #print( "Back from...progress_infections()" )

        # Update immune agents
        def progress_immunities():
            # immunity timer: decrement for each immune person
            # immunity flag: clear for each new sus person
            cursor.execute("UPDATE agents SET immunity_timer = (immunity_timer-1) WHERE immunity=1 AND immunity_timer>0" )
            cursor.execute("UPDATE agents SET immunity = 0 WHERE immunity = 1 AND immunity_timer=0" )
        def progress_immunities_np():
            condition = np.logical_and(data['immunity'], data['immunity_timer'] > 0)
            data['immunity_timer'][condition] -= 1
            condition = np.logical_and(data['immunity'], data['immunity_timer'] == 0)
            data['immunity'][condition] = False

        progress_immunities_np()
        #print( "Back from...progress_immunities()" )

        # We want to count the number of incubators by now all at once not in a for loop.
        node_counts_incubators = np.bincount( data['node'][data['incubation_timer']>=1] )
        if len( node_counts_incubators ) == 0:
            print( "node_counts_incubators came back size 0." )
            node_counts_incubators = np.zeros( settings.num_nodes )
            #raise ValueError( "node_counts_incubators came back size 0." )
        sorted_items = sorted(currently_infectious.items())
        inf_np = np.array([value for _, value in sorted_items])
        foi = (inf_np-node_counts_incubators) * base_infectivity
        sus_np = np.array(list(currently_sus.values()))
        new_infections = (foi * sus_np).astype(int)
        #print( new_infections )

        def handle_transmission( node=0 ):
            # Step 5: Update the infected flag for NEW infectees
            def handle_new_infections_np(new_infections):
                #print( f"We are doing transmission to {new_infections} in node {node}." )
                # Create a boolean mask based on the conditions in the subquery
                subquery_condition = np.logical_and(~data['infected'], ~data['immunity'])
                subquery_condition = np.logical_and(subquery_condition, (data['node'] == node))
                # BUG: The above is returning True for all nodes, not just agents with the current node!

                # Get the indices of eligible agents using the boolean mask
                eligible_agents_indices = np.where(subquery_condition)[0]

                # Randomly sample 'new_infections' number of indices
                selected_indices = np.random.choice(eligible_agents_indices, size=min(new_infections, len(eligible_agents_indices)), replace=False)

                # Update the 'infected' column based on the selected indices
                data['infected'][selected_indices] = True
                data['incubation_timer'][selected_indices] = 2

            #print( new_infections )
            if new_infections[node]>0:
                handle_new_infections_np(new_infections[node])
                #print( "Back from new_infections_np" )

            #print( f"{new_infections} new infections in node {node}." )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(handle_transmission, settings.nodes))
            #results = list(executor.map(handle_new_infections_np, settings.nodes))

        #for node in settings.nodes:
            #handle_transmission( node )
            #print( "Back from...handle_transmission()" )
            # handle new infectees, set new infection timer
            #cursor.execute( "UPDATE agents SET infection_timer=FLOOR(4+10*(RANDOM() + 9223372036854775808)/18446744073709551616) WHERE infected AND infection_timer=0" )
        condition = np.logical_and(data['infected'], data['infection_timer'] == 0)
        data['infection_timer'][condition] = np.random.randint(4, 15, size=np.sum(condition))
        #print( "Back from...init_inftimers()" )

        def migrate():
            if timestep % 7 == 0: # every week
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
                    ''', { 'max_node': settings.num_nodes-1 } )
        def migrate_np():
            if timestep % 2 == 0: # every week
                #print( "It's Sunday, migrate." )
                infected = np.where( data['infected'] )[0]
                fraction = int(len(infected)*0.01)
                selected = np.random.choice( infected, fraction )
                # Update the 'nodes' array based on the specified conditions
                data['node'][selected] = np.where(data['node'][selected] - 1 < 0, settings.num_nodes - 1, data['node'][selected] - 1 )

        #migrate():
        migrate_np()
        #print( "Back from migrate_np" )

        #conn.commit()
        #print( "Back from...commit()" )
        #print( f"{cursor.execute('select * from agents where infected limit 25').fetchall()}".replace("), ",")\n") )
        print( "*" )
        currently_infectious, currently_sus = report( data, timestep, csvwriter )



    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    data = load()

    # Create a CSV file for reporting
    csvfile = open('simulation_report.csv', 'w', newline='') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'Recovered'])

    # Run the simulation for 1000 timesteps
    run_simulation(data, csvwriter, num_timesteps=settings.duration )

