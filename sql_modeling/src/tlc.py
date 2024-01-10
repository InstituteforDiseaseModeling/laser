# Import a model
#import sir_sql as model
#import sir_sql_polars as model
import sir_numpy as model

import settings
import report

report.write_report = True # sometimes we want to turn this off to check for non-reporting bottlenecks


def run_simulation(ctx, csvwriter, num_timesteps):
    currently_infectious, currently_sus, cur_reco = model.collect_report( ctx )
    report.write_timestep_report( csvwriter, 0, currently_infectious, currently_sus, cur_reco )

    for timestep in range(1, num_timesteps + 1):

        # We almost certainly won't waste time updating everyone's ages every timestep but this is 
        # here as a placeholder for "what if we have to do simple math on all the rows?"
        ctx = model.update_ages( ctx )

        # We should always be in a low prev setting so this should only really ever operate
        # on ~1% of the active population
        ctx = model.progress_infections( ctx )

        # The perma-immune should not consume cycles but there could be lots of waning immune
        ctx = model.progress_immunities( ctx )

        # The core transmission part begins
        new_infections = model.calculate_new_infections( ctx, currently_infectious, currently_sus )

        # TBD: for loop should probably be implementation-specific
        ctx = model.handle_transmission( ctx, new_infections )

        ctx = model.add_new_infections( ctx )

        # Transmission is done, now migrate some. Only infected?
        ctx = model.migrate( ctx, timestep, num_infected=sum(currently_infectious.values()) )
        #conn.commit() # deb-specific

        # Report
        currently_infectious, currently_sus, cur_reco = model.collect_report( ctx )
        report.write_timestep_report( csvwriter, timestep, currently_infectious, currently_sus, cur_reco )

    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    # Initialize the 'database' (or load the dataframe/csv)
    # ctx might be db cursor or dataframe or dict of numpy vectors
    ctx = model.initialize_database()

    csv_writer = report.init()

    # Run the simulation for 1000 timesteps
    run_simulation(ctx, csv_writer, num_timesteps=settings.duration )

