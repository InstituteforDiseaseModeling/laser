import sys
import os
sys.path.insert(0, os.getcwd())

import pdb

# Import a model
import sir_numpy_c as model
import numpy as np

import settings
import demographics_settings
import report

#report.write_report = True # sometimes we want to turn this off to check for non-reporting bottlenecks
report_births = []

new_infections_empty = {}
for i in range(demographics_settings.num_nodes):
    new_infections_empty[ i ] = 0

def collect_and_report(csvwriter, timestep, ctx):
    currently_infectious, currently_sus, cur_reco, totals = model.collect_report( ctx )
    counts = {
            "S": currently_sus,
            "I": currently_infectious,
            "R": cur_reco 
        }
    #print( f"Counts =\nS:{counts['S']}\nI:{counts['I']}\nR:{counts['R']}" )

    try:
        report.write_timestep_report( csvwriter, timestep, counts["I"], counts["S"], counts["R"], new_births=report_births )
    except Exception as ex:
        raise ValueError( f"Exception {ex} at timestep {timestep} and counts {counts['I']}, {counts['S']}, {counts['R']}" )
    return counts, totals

def run_simulation(ctx, csvwriter, num_timesteps, sm=-1, bi=-1, mf=-1):
    counts, totals = collect_and_report(csvwriter,0, ctx)
    if sm==-1:
        sm = settings.seasonal_multiplier
    if bi==-1:
        bi = settings.base_infectivity
    if mf==-1:
        mf = settings.migration_fraction

    for timestep in range(1, num_timesteps + 1):
        # We should always be in a low prev setting so this should only really ever operate
        # on ~1% of the active population
        ctx = model.progress_infections( ctx )

        # The perma-immune should not consume cycles but there could be lots of waning immune
        ctx = model.progress_immunities( ctx )

        # The core transmission part begins
        if timestep>settings.burnin_delay:
            new_infections = list()
            if sum( counts["I"].values() ) > 0:
                new_infections = model.calculate_new_infections( ctx, counts["I"], counts["S"], totals, timestep, seasonal_multiplier=sm, base_infectivity=bi )
                report.new_infections = new_infections 
            #print( f"new_infections=\n{new_infections}" )

            # TBD: for loop should probably be implementation-specific
            if sum( new_infections ) > 0:
                ctx = model.handle_transmission( ctx, new_infections, counts["S"] )
                ctx = model.add_new_infections( ctx )

            ctx = model.distribute_interventions( ctx, timestep )

        # Transmission is done, now migrate some. Only infected?
        if timestep>settings.burnin_delay and settings.num_nodes>1 and mf>0:
            ctx = model.migrate( ctx, timestep, migration_fraction=mf )

        # if we have had total fade-out, inject imports
        if timestep>settings.burnin_delay and sum(counts["I"].values()) == 0 and settings.import_cases > 0:
            def divide_and_round(susceptibles):
                for node, count in susceptibles.items():
                    susceptibles[node] = round(count / 80)
                return list(susceptibles.values())
            import_cases = np.array(divide_and_round( counts["S"] ), dtype=np.uint32)
            print( f"ELIMINATION Detected: Reseeding: Injecting new cases." )
            model.handle_transmission( ctx, import_cases, counts["S"] )
            

        # We almost certainly won't waste time updating everyone's ages every timestep but this is 
        # here as a placeholder for "what if we have to do simple math on all the rows?"
        global report_births, report_deaths 
        ( report_births, report_deaths ) = model.update_ages( ctx, totals, timestep )

        # Report
        counts, totals = collect_and_report(csvwriter,timestep,ctx)
        

    print(f"Simulation completed. Report saved to '{settings.report_filename}'.")

# Main simulation
if __name__ == "__main__":
    # Initialize the 'database' (or load the dataframe/csv)
    # ctx might be db cursor or dataframe or dict of numpy vectors
    ctx = model.initialize_database()
    ctx = model.eula_init( ctx, demographics_settings.eula_age )

    csv_writer = report.init()

    # Run the simulation for 1000 timesteps
    from functools import partial
    runsim = partial( run_simulation, ctx=ctx, csvwriter=csv_writer, num_timesteps=settings.duration )
    from timeit import timeit
    runtime = timeit( runsim, number=1 )
    print( f"Execution time = {runtime}." )

    # The lines below are required for calibration; running them imposes requirements, and uses more time and memory.
    #import post_proc
    #post_proc.analyze()

