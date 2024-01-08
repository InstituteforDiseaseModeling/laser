import polars as pl
import random
import csv
import numpy as np 
from sparklines import sparklines
import time
import pdb
from settings import * # local file
import settings

settings.base_infectivity = 0.00001
start_timer = time.time()
last_time = time.time()

def load( pop_file ):
    # Replace 'your_file.csv' with the actual path to your CSV file
    df = pl.read_csv( pop_file )

    settings.pop = df.shape[0]
    print( f"Population={settings.pop}" )
    #settings.nodes = [ node for node in np.unique(columns['node']) ]
    settings.num_nodes = pl.Series.n_unique(df['node'])
    settings.nodes = [ node for node in range(settings.num_nodes) ]
    print( f"Nodes={settings.num_nodes}" )
    # Now 'columns' is a dictionary where keys are column headers and values are NumPy arrays
    return df

def report( df, timestep, csvwriter ):
    #print( "Start report." )
    # Count agents in each state
    ctx = pl.SQLContext(agents=df, eager_execution=True)
    results = ctx.execute('SELECT node, COUNT(*) FROM agents WHERE infected=0 AND immunity=0 GROUP BY node')

    susceptible_counts = dict(zip(results['node'],results['count']))
    # query gives sparse results. De-sparsify (aka pad)
    for node in settings.nodes:
        if node not in susceptible_counts:
            susceptible_counts[node] = 0

    results = ctx.execute('SELECT node, COUNT(*) FROM agents WHERE infected=1 GROUP BY node')
    infected_counts = dict(zip(results['node'],results['count']))
    for node in settings.nodes:
        if node not in infected_counts:
            infected_counts[node] = 0

    results = ctx.execute('SELECT node, COUNT(*) FROM agents WHERE immunity=1 GROUP BY node')
    recovered_counts = dict(zip(results['node'],results['count']))
    for node in settings.nodes:
        if node not in recovered_counts:
            recovered_counts[node] = 0

    # Write the counts to the CSV file
    #print( f"T={timestep}, S={susceptible_counts}, I={infected_counts}, R={recovered_counts}" )
    print( f"T={timestep}" )
    print( list( sparklines( infected_counts.values() ) ) )
    for node in settings.nodes:
        csvwriter.writerow([timestep,
            node,
            susceptible_counts[node] if node in susceptible_counts else 0,
            infected_counts[node] if node in infected_counts else 0,
            recovered_counts[node] if node in recovered_counts else 0,
            ]
        )
    #print( "Stop report." )
    global last_time
    print( f"Elapsed Time for timestep: {(time.time()-last_time):0.2f}" )
    last_time = time.time()
    return infected_counts, susceptible_counts

# Function to run the simulation for a given number of timesteps
def run_simulation(df, csvwriter, num_timesteps):
    import timeit
    currently_infectious, currently_sus = report( df, 0, csvwriter )

    for timestep in range(1, num_timesteps + 1):
        # Update infected agents
        # infection timer: decrement for each infected person

        def update_ages( df ):
            #cursor.execute("UPDATE agents SET age = age+1/365")
            df = df.with_columns(age=pl.col("age") + 1/365)
            return df
        df = update_ages( df )
        #print( "Back from...update_ages()" )
        #print( f"update_ages took {age_time}" )

        def progress_infections(df):
            # Progress Existing Infections
            # Clear Recovereds
            # infected=0, immunity=1, immunity_timer=30-ish
            df = df.with_columns(infection_timer=pl.when( pl.col("infection_timer")>1 ).then(pl.col("infection_timer") - 1).otherwise( 0 ) )
            df = df.with_columns(incubation_timer=pl.when( pl.col("incubation_timer")>1 ).then(pl.col("incubation_timer") - 1).otherwise( 0 ) )
            df = df.with_columns( newly_cleared=pl.when( (
                        (pl.col("infected") == 1 ) &
                        (pl.col("infection_timer") == 0))
                    ).then(True).otherwise(False)
                )
            # The check for newly_cleared is never returning anything even though the query above shows it should
            df = df.with_columns( infected=pl.when( pl.col( "newly_cleared" ) ).then(0).otherwise( pl.col( "infected" ) ) )
            df = df.with_columns( immunity=pl.when( pl.col( "newly_cleared" ) ).then(1).otherwise( pl.col( "immunity" ) ) )
            df = df.with_columns( immunity_timer=pl.when( pl.col( "newly_cleared" ) == 1 ).then(
                pl.lit( np.random.randint(
                    10,40, size=df.height
                ) ) ).otherwise(
                    pl.col( "immunity_timer"
                ) ) ) # TBD: Needs to be from distribution
            df = df.drop( "newly_cleared" )
            ctx = pl.SQLContext(agents=df, eager_execution=True)
            results = ctx.execute('SELECT node, immunity_timer FROM agents WHERE immunity_timer>0')

            return df

        df = progress_infections(df)
        #print( "Back from...progress_infections()" )

        # Update immune agents
        def progress_immunities(df):
            # immunity timer: decrement for each immune person
            # immunity flag: clear for each new sus person
            #cursor.execute("UPDATE agents SET immunity_timer = (immunity_timer-1) WHERE immunity=1 AND immunity_timer>0" )
            #pdb.set_trace()
            df = df.with_columns( immunity_timer=pl.when( 
                    (pl.col( "immunity" ) == 1 ) &
                    (pl.col( "immunity_timer" ) > 0 ) 
                ).then(
                    pl.col( "immunity_timer" ) - 1
                ).otherwise(
                    pl.col( "immunity_timer" )
                ) )
            #cursor.execute("UPDATE agents SET immunity = 0 WHERE immunity = 1 AND immunity_timer=0" )
            df = df.with_columns( immunity=pl.when( 
                    (pl.col( "immunity" ) == 1 ) &
                    (pl.col( "immunity_timer" ) == 0 ) 
                ).then(
                    0
                ).otherwise(
                    pl.col( "immunity" )
                ) )
            return df
        df = progress_immunities(df)
        #print( "Back from...progress_immunities()" )

        def handle_transmission( df, node=0 ):
            if new_infections[node] == 0:
                return df
            # We have total susceptibles by node, but not the individuals
            condition = (pl.col("infected") == 0) & (pl.col("immunity") == 0) & (pl.col("node") == node)
            if new_infections[ node ] > len( df.filter(condition) ):
                print( f"WARNING: More new infections requested {new_infections[node]} than susceptibles {len(df.filter(condition))}!" )
                pdb.set_trace()
            sampled_df = df.filter(condition).sample( new_infections[ node ] )
            # Should we add a newly_infected=True col/flag for these guys so make them easy to catch?
            sampled_df = sampled_df.with_columns( infected=1 )
            sampled_df = sampled_df.with_columns( incubation_timer=2 )
            def get_infection_timer_init():
                return random.randint( 4,10 )
            #sampled_df = sampled_df.with_columns( infection_timer=get_infection_timer_init() ) # should be a draw
            sampled_df = sampled_df.with_columns( infection_timer=pl.lit(np.random.randint(
                4,10, size= sampled_df.height
            ) ) )
            # We should set their infection timer here or globally, not do incubation_timer here 
            # and infection timer globally
            # Infect: CONVERT: data['infected'][selected_indices] = True
            # Set incubation timer: CONVERT: data['incubation_timer'][selected_indices] = 2
            df = df.update( sampled_df, left_on="id", right_on="id", how="outer" )
            #print( f"{new_infections[node]} new infections in node {node}." )
            return df

        # We want to get a dict of node ids to number of new_infections to create...
        filtered_df = df.filter(pl.col("incubation_timer") >= 1)
        node_counts_incubators = np.zeros( settings.num_nodes )
        results = ( filtered_df.groupby( "node" ).agg( pl.col( "node" ).count().alias( "count" )).sort( "node" ) )
        node_counts_incubators2 = dict(zip(results['node'],results['count']))
        for node in node_counts_incubators:
            if int(node) in node_counts_incubators2:
                node_counts_incubators += node_counts_incubators2[int(node)]
        if len( node_counts_incubators ) == 0:
            print( "node_counts_incubators came back size 0." )
            #raise ValueError( "node_counts_incubators came back size 0." )
        sorted_items = sorted(currently_infectious.items())
        inf_np = np.array([value for _, value in sorted_items])
        foi = (inf_np-node_counts_incubators) * settings.base_infectivity
        # foi should be more of a probability of getting infected but we're being simplistic for now. Cap at 1.
        foi = np.array([ min(1,x) for x in foi ])
        sus_np = [value for key, value in sorted(currently_sus.items())]
        new_infections = (foi * sus_np).astype(int)

        for node in settings.nodes:
            df = handle_transmission( df, node )
            #print( "Back from...handle_transmission()" )

        def migrate( df ):
            # 1% weekly migration ought to cause infecteds from seed node to move to next node
            if timestep % 7 == 0: # every week (or day, depending on what I've set it to)
                condition = (pl.col("infected") == 1)
                sampled_df = df.filter(condition).sample( int( sum(currently_infectious.values()) * 0.01 ) )
                sampled_df = sampled_df.with_columns(pl.when( pl.col("node")>0 ).then(pl.col("node") - 1).otherwise( settings.num_nodes-1 ) )
                df = df.update( sampled_df, left_on="id", right_on="id", how="outer" )
            return df
        df = migrate( df )
        
        #print( "*****" )
        currently_infectious, currently_sus = report( df, timestep, csvwriter )



    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    # Create a CSV file for reporting
    csvfile = open('simulation_report.csv', 'w', newline='') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'Recovered'])
    df = load( settings.pop_file )

    # Run the simulation for 1000 timesteps
    run_simulation(df, csvwriter, num_timesteps=duration )

