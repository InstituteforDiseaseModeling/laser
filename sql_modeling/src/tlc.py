import pdb
# Import a model
#import sir_sql as model
#import sir_mysql as model
#import sir_sql_polars as model
import sir_numpy as model
#import sir_numpy_c as model
from copy import deepcopy

import settings
import report

report.write_report = True # sometimes we want to turn this off to check for non-reporting bottlenecks
fractions = {}

def collect_and_report(csvwriter, timestep):
    currently_infectious, currently_sus, cur_reco = model.collect_report( ctx )
    counts = {
            "S": deepcopy( currently_sus ),
            "I": deepcopy( currently_infectious ),
            "R": deepcopy( cur_reco ) 
        }
    #print( f"Counts =\nS:{counts['S']}\nI:{counts['I']}\nR:{counts['R']}" )
    def normalize( sus, inf, rec ):
        totals = {}
        for idx in currently_sus.keys():
            totals[ idx ] = sus[ idx ] + inf[ idx ] + rec[ idx ]
            sus[ idx ] /= totals[ idx ] 
            inf[ idx ] /= totals[ idx ] 
            rec[ idx ]/= totals[ idx ] 
        return totals
    totals = normalize( currently_sus, currently_infectious, cur_reco )
    fractions = {
            "S": currently_sus,
            "I": currently_infectious,
            "R": cur_reco 
        }
    #print( fractions["S"][10] )
    #print( counts["S"][10] )
    report.write_timestep_report( csvwriter, timestep, counts["I"], counts["S"], counts["R"] )
    return counts, fractions, totals

def run_simulation(ctx, csvwriter, num_timesteps):
    counts, fractions, totals = collect_and_report(csvwriter,0)
    for timestep in range(1, num_timesteps + 1):

        # We should always be in a low prev setting so this should only really ever operate
        # on ~1% of the active population
        ctx = model.progress_infections( ctx )

        # The perma-immune should not consume cycles but there could be lots of waning immune
        ctx = model.progress_immunities( ctx )

        # The core transmission part begins
        new_infections = model.calculate_new_infections( ctx, fractions["I"], fractions["S"], totals )
        #print( f"new_infections=\n{new_infections}" )

        # TBD: for loop should probably be implementation-specific
        ctx = model.handle_transmission( ctx, new_infections )

        ctx = model.add_new_infections( ctx )

        ctx = model.distribute_interventions( ctx, timestep )
        # Transmission is done, now migrate some. Only infected?
        ctx = model.migrate( ctx, timestep, num_infected=sum(fractions["I"].values()) )

        # We almost certainly won't waste time updating everyone's ages every timestep but this is 
        # here as a placeholder for "what if we have to do simple math on all the rows?"
        ctx = model.update_ages( ctx, totals )

        # Report
        #currently_infectious, currently_sus, cur_reco = model.collect_report( ctx )
        #totals = normalize( currently_sus, currently_infectious, cur_reco )
        counts, fractions, totals = collect_and_report(csvwriter,timestep)
        #report.write_timestep_report( csvwriter, timestep, currently_infectious, currently_sus, cur_reco )

    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    # Initialize the 'database' (or load the dataframe/csv)
    # ctx might be db cursor or dataframe or dict of numpy vectors
    ctx = model.initialize_database()
    #ctx = model.init_db_from_csv( settings )
    ctx = model.eula( ctx, 15, eula_strategy="downsample" )

    csv_writer = report.init()

    # Run the simulation for 1000 timesteps
    from timeit import timeit
    from functools import partial
    runsim = partial( run_simulation, ctx=ctx, csvwriter=csv_writer, num_timesteps=settings.duration )
    runtime = timeit( runsim, number=1 )
    print( f"Execution time = {runtime}." )

