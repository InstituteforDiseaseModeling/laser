import random
import csv
import numpy as np
import concurrent.futures
from functools import partial
import pdb

import settings
import report
from model_numpy import eula



def load( pop_file ):
    """
    Load population from csv file as np arrays. Each property column is an np array.
    """
    # Load the entire CSV file into a NumPy array
    header_row = np.genfromtxt(pop_file, delimiter=',', dtype=str, max_rows=1)

    # Load the remaining data as numerical values, skipping the header row
    data = np.genfromtxt(pop_file, delimiter=',', dtype=float, skip_header=1)

    # Extract headers from the header row
    headers = header_row

    # Load each column into a separate NumPy array
    columns = {header: data[:, i] for i, header in enumerate(headers)}
    columns['infected'] = columns['infected'].astype(bool)
    columns['immunity'] = columns['immunity'].astype(bool)
    columns['node'] = columns['node'].astype(np.uint32)
    columns['infection_timer'] = columns['infection_timer'].astype(np.float32) # int better?
    columns['incubation_timer'] = columns['incubation_timer'].astype(np.float32) # int better?
    columns['immunity_timer'] = columns['immunity_timer'].astype(np.float32) # int better?
    columns['age'] = columns['age'].astype(np.float32)
    columns['expected_lifespan'] = columns['expected_lifespan'].astype(np.float32)
    if "mcw" in columns:
        columns.pop( "mcw" )

    settings.pop = len(columns['infected'])
    print( f"Population={settings.pop}" )

    add_expansion_slots( columns,num_slots=50000 )
    # Pad with a bunch of zeros
    return columns

def add_expansion_slots( columns, num_slots=10000 ):
    num_slots = int(num_slots)
    print( f"Adding {num_slots} expansion slots for future babies." )
    new_ids = [ x for x in range( num_slots ) ]
    new_nodes = np.ones( num_slots, dtype=np.uint32 )*-1
    new_ages = np.ones( num_slots, dtype=np.float32 )*-1
    new_infected = np.zeros( num_slots, dtype=bool )
    new_immunity = np.zeros( num_slots, dtype=bool )
    new_immunity_timer = np.zeros( num_slots ).astype( np.float32 )
    new_infection_timer = np.zeros( num_slots ).astype( np.float32 )
    new_incubation_timer = np.zeros( num_slots ).astype( np.float32 )
    new_expected_lifespan = np.ones( num_slots, dtype=np.float32 )*-1

    settings.nodes = [ node for node in np.unique(columns['node']) ]
    settings.num_nodes = len(settings.nodes)
    print( f"Nodes={settings.num_nodes}" )
    # Now 'columns' is a dictionary where keys are column headers and values are NumPy arrays

    columns['id'] = np.concatenate((columns['id'], new_ids))
    columns['node'] = np.concatenate((columns['node'], new_nodes)).astype( np.uint32 )
    columns['age'] = np.concatenate((columns['age'], new_ages))
    columns['infected'] = np.concatenate((columns['infected'], new_infected))
    columns['infection_timer'] = np.concatenate((columns['infection_timer'], new_infection_timer))
    columns['incubation_timer'] = np.concatenate((columns['incubation_timer'], new_incubation_timer))
    columns['immunity'] = np.concatenate((columns['immunity'], new_immunity))
    columns['immunity_timer'] = np.concatenate((columns['immunity_timer'], new_immunity_timer))
    columns['expected_lifespan'] = np.concatenate((columns['expected_lifespan'], new_expected_lifespan))

def initialize_database():
    return load( settings.pop_file )
    
def eula_init( df, age_threshold_yrs = 5, eula_strategy=None ):
    # Create a boolean mask for elements to keep
    def filter_strategy():
        # test out what happens if we render big chunks of the population epi-borrowing
        condition = np.logical_and(~columns['infected'], columns['age']>age_threshold_yrs )
        columns['immunity'][condition] = 1
        columns['immunity_timer'][condition] = -1

    def purge_strategy():
        # Note this is just for testing; for real work we need to keep track of our total pop
        mask = (df['age'] <= age_threshold_yrs) | (df['infected'] != 0)
        for column in df.keys():
            df[column] = df[column][mask]

    def downsample_strategy():
        # mask = (df['age'] <= age_threshold_yrs) | (df['infected'] != 0)
        filter_arr = df['age']>=0
        # For permanently recovereds, we want those over threshold age and not infected
        mask = ((df['age'] >= age_threshold_yrs) & (~df['infected']))[filter_arr]
        # need this by node
        # BUG! This seems to be returning the number of susceptibles, not recovereds

        # For actual removal, we want thsoe not infected and those over threshold age but don't remove those with age == -1
        # So keep the rest
        mask = ((df['age'] < age_threshold_yrs) | (df['infected'] ))

        # We are going to delete number_recovereds agents
        # Then we're going to add 1 with a huge mcw: Not any more
        # TBD: Recycle 1 of the deletes instead of delete and add
        # Add 1 perma-immune, mega-agent per node...
        # New plan: Add 1 perma-immune mega-agent per age per node...
        eula.init()

        for column in df.keys():
            df[column] = df[column][mask]


    print( "Ignoring requested strategy; using downsample only for now." )
    downsample_strategy()
    return df

def collect_report( data ):
    """
    Report data to file for a given timestep.
    """
    #print( "Start timestep report." )
    # THIS IS MESSED UP BUT I WASTED AN HOUR ON THE ALTERNATIVE!!!
    condition_mask = np.logical_and(~data['infected'], ~data['immunity'])
    unique_nodes, counts = np.unique(data['node'][condition_mask], return_counts=True)

    # Display the result
    susceptible_counts_db = list(zip(unique_nodes, counts))
    susceptible_counts = {values[0]: values[1] for idx, values in enumerate(susceptible_counts_db)}
    for node in settings.nodes:
        if node not in susceptible_counts:
            susceptible_counts[node] = 0

    # Because we put dead people in "purgatory"...
    if 4294967295 in susceptible_counts.keys(): # uint32(-1)
        susceptible_counts.pop(4294967295)
    if -1 in susceptible_counts.keys(): 
        susceptible_counts.pop(-1)
    if len(susceptible_counts) > len(settings.nodes):
        pdb.set_trace()
        raise ValueError( f"Too many susceptible nodes." )

    unique_nodes, counts = np.unique(data['node'][data['infected']], return_counts=True)
    infected_counts_db = list(zip(unique_nodes, counts))
    infected_counts = {values[0]: values[1] for idx, values in enumerate(infected_counts_db)}
    for node in settings.nodes:
        if node not in infected_counts:
            infected_counts[node] = 0
    if len(infected_counts) > len(settings.nodes):
        pdb.set_trace()
        raise ValueError( f"Too many infected nodes." )

    def count_recos( node, immunity ):
        # Boolean indexing to filter rows where immunity is 1
        filtered_rows = (immunity==1)

        # Use the filtered rows to get the corresponding node
        filtered_node = node[filtered_rows]

        # Use numpy.bincount to calculate the sum of Rs for each node
        unique_nodes, counts = np.unique(filtered_node, return_counts=True)
        counts_by_node = dict(zip(unique_nodes, counts))

        # Display the result
        # Does this code assume there's an entry for each node even if the total is 0?
        #pdb.set_trace()
        recovered_counts_eula = eula.get_recovereds_by_node()
        for node in range( settings.num_nodes ):
            # Now add in eulas
            if node not in counts_by_node:
                counts_by_node[node] = 0
            #counts_by_node[node] += sum(recovered_counts_eula[node].values())
            counts_by_node[node] += recovered_counts_eula[node]
        return counts_by_node

    recovered_counts = count_recos( data['node'], data['immunity'] )

    #print( "Stop timestep report." )
    return infected_counts, susceptible_counts, recovered_counts 


def update_ages( data, totals ):
    """
    import numba
    @numba.jit(parallel=True,nopython=True)
    def update_ages_nb( ages ):
        n = len(ages)
        for i in range(n):
            ages[i] += 1/365
        return ages
        #data['age'] += 1/365
    """
    def update_ages_np( ages ):
        ages[ages>0] += 1/365
        return ages

    data['age'] = update_ages_np( data['age'] )
    # data = 
    births( data, totals )
    #data = 
    deaths( data )
    return data

def births_from_cbr( node_pops, rate=30 ):
    # TBD: births = CBR & node_pop / 1000
    # placeholder: just say 10 per node for now to test rest of code path
    new_babies = {}
    for node in node_pops:
        cbr_node = settings.fertility_interval * rate * (node_pops[node]/1000.0)/365.0
        new_babies[node] = np.random.poisson( cbr_node )
    return new_babies 
  
def births(data,totals_by_node):
    # Births
    # 1) demographic_dependent_Rate: 
    # Calculate number of women of child-bearing age: constant across nodes
    # Add new babies as percentage of that.
    # Can't do demographic_dependent_Rate if downsampling recovereds to N, hence:
    # Or CBR
    # totals_by_node supports CBR

    def births_from_demog():
        # Return actual numbers of new babies by node, not just rates
        # Create a boolean mask for the age condition
        age_condition_mask = (data['age'] > 15) & (data['age'] < 45)

        # Filter data based on the age condition
        filtered_nodes = data['node'][age_condition_mask]
        filtered_ages  = data['age'][age_condition_mask]

        # Calculate the count of data per node and divide by 2
        unique_nodes, counts = np.unique(filtered_nodes, return_counts=True)
        wocba = np.column_stack((unique_nodes, counts // 2))

        # Create a dictionary from the wocba array
        wocba_dict = dict(wocba)
        num_new_babies = dict()
        for node, count in wocba_dict.items():
            num_new_babies[node] = np.sum(np.random.rand(count) < 2.7e-4)
        return num_new_babies


    # Function to add newborns
    def add_newborns( nodes ):
        babies = len(nodes)
        # Generate newborn data
        #last_id = data['id'][-1]
        # find an entry with age==-1 to use, or find a bunch
        indices = np.where( data['age'] == -1 )[0][:babies]
        #new_ids = np.arange(last_id + 1, last_id + 1 + babies)
        #new_ids = data['id'][indices][:babies]
        new_nodes = np.full(babies, nodes)
        new_ages = np.zeros(babies)
        new_infected = np.full(babies,False)
        new_infection_timer = np.zeros(babies)
        new_incubation_timer = np.zeros(babies)
        new_immunity = np.full(babies,False)
        new_immunity_timer = np.zeros(babies)
        new_expected_lifespan = np.random.normal(loc=75, scale=7, size=babies).astype(np.float32)

        def reincarnate( data, indices, new_nodes, new_ages, new_infected, new_infection_timer, new_incubation_timer, new_immunity, new_immunity_timer, new_expected_lifespan ):
            # This is memory-smarter option where we recycle agents
            # TBD: Make c version
            data['node'][indices] = new_nodes
            data['age'][indices] = new_ages
            data['infected'][indices] = new_infected
            data['infection_timer'][indices] = new_infection_timer
            data['incubation_timer'][indices] = new_incubation_timer
            data['immunity'][indices] = new_immunity
            data['immunity_timer'][indices] = new_immunity_timer
            data['expected_lifespan'][indices] = new_expected_lifespan
        # We have selected all the indices of unborn babies (age=-1) to birth.
        reincarnate( data, indices, new_nodes, new_ages, new_infected, new_infection_timer, new_incubation_timer, new_immunity, new_immunity_timer, new_expected_lifespan )

    new_babies = births_from_cbr( totals_by_node, rate=settings.cbr )
    #print( f"New babies by node: {new_babies}" )
    # Iterate over nodes and add newborns
    #for node, count in new_babies.items():
    #    if count > 0:
    #        add_newborns(node, count)
    keys = np.array(list(new_babies.keys()))
    values = np.array(list(new_babies.values()))

    # Create the numpy array by repeating each key according to its corresponding value
    result_array = np.repeat(keys, values)
    add_newborns( result_array )

    return data

def deaths(data):
    # Non-disease deaths
    # Create a boolean mask for the deletion condition
    def old_style():
        delete_mask = (data['age'] >= data['expected_lifespan']) & (data['age']>=0)
        if np.count_nonzero( delete_mask ):

            #data['infected'] = np.delete( data['infected'], np.where( delete_mask ) )
            #data[col] = np.delete( data[col], np.where( delete_mask ) )
            #data[col] = data[col][~delete_mask]
            data['node'][delete_mask]  = -1
            data['age'][delete_mask] = -1
            data['infected'][delete_mask] = 0
            data['immunity'][delete_mask] = 0
            data['infection_timer'][delete_mask] = 0
            data['immunity_timer'][delete_mask] = 0
            data['incubation_timer'][delete_mask] = 0
            data['expected_lifespan'][delete_mask] = -1

            print( f"{np.count_nonzero(delete_mask)} new deaths." )
    
    def eula_death_by_rate():
        eula.progress_natural_mortality()

    eula_death_by_rate()
    return data

def update_births_deaths( data ):
    data = deaths(data)
    data = births(data)
    return data

def progress_infections( data ):
    # Update infected agents
    # infection timer: decrement for each infected person
    data['infection_timer'][data['infection_timer'] >= 1] -= 1
    data['incubation_timer'][data['incubation_timer'] >= 1] -= 1
    # some people clear
    condition = np.logical_and(data['infected'], data['infection_timer'] == 0)
    data['infected'][condition] = False
    # recovereds gain immunity
    data['immunity_timer'][condition] = np.random.randint(10, 41, size=np.sum(condition))
    data['immunity'][condition] = True

    return data

# Update immune agents
def progress_immunities( data ):
    # immunity decays
    condition = np.logical_and(data['immunity'], data['immunity_timer'] > 0)
    data['immunity_timer'][condition] -= 1
    # Recoverd->Susceptible
    condition = np.logical_and(data['immunity'], data['immunity_timer'] == 0)
    data['immunity'][condition] = False

    return data

def calculate_new_infections( data, inf, sus, totals ):
    # We are currently only passing in inf and sus fractions and totals. Not passing in incubators.
    # We want to count the number of incubators by now all at once not in a for loop.
    node_counts_incubators = np.zeros(len(inf))
    node_counts_incubators = np.bincount( data['node'][data['age']>=0], weights=(data['incubation_timer']>=1)[data['age']>=0] )
    #exposed_fraction = {}
    for idx in range(len(node_counts_incubators)):
        exposed_fraction = node_counts_incubators[idx]/totals[idx]
        inf[idx] -= exposed_fraction 
        #print( f"infectious fraction for node {idx} = {inf[idx]} after subtracting {node_counts_incubators[idx]} incubators." )
    ret_ni = np.zeros(settings.num_nodes).astype( np.uint32 )
    for node in range(settings.num_nodes):
        ret_ni[node] = int(sus[node]*inf[node]*settings.base_infectivity)
    #print( f"New Infections: {new_infections} = {np.array(sorted(sus.values()))} * {np.array(sorted(inf.values()))} * {settings.base_infectivity}" )
    return ret_ni

def handle_transmission_by_node( data, new_infections, node=0 ):
    def handle_new_infections(new_infections):
        # print( f"We are doing transmission to {new_infections} in node {node}." )
        # Create a boolean mask based on the conditions in the subquery
        subquery_condition = np.logical_and(~data['infected'], ~data['immunity'])
        subquery_condition = np.logical_and(subquery_condition, (data['node'] == node))

        # Get the indices of eligible agents using the boolean mask
        eligible_agents_indices = np.where(subquery_condition)[0]

        # Randomly sample 'new_infections' number of indices
        selected_indices = np.random.choice(eligible_agents_indices, size=min(new_infections, len(eligible_agents_indices)), replace=False)

        # Update the 'infected' column based on the selected indices
        #if any(data['infected'][selected_indices]):
            #print( "ERROR: Infecting someone already infected!." )
        data['infected'][selected_indices] = True
        #data['incubation_timer'][selected_indices] = 2

    #print( new_infections[node] )
    if new_infections[node]>0:
        handle_new_infections(new_infections[node])

    #print( f"{new_infections} new infections in node {node}." )
    return data

def handle_transmission( data_in, new_infections_in ):
    # We want to do this in parallel;
    htbn = partial( handle_transmission_by_node, data_in, new_infections_in )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(htbn, settings.nodes))
    return data_in

def add_new_infections( data ):
    # Actually this just sets the new infection timers (globally) for all the new infections
    # New infections themselves are set node-wise
    def add_new_infections_np( data ):
        condition = np.logical_and(data['infected'], data['infection_timer'] == 0)
        min_infection_dur = 11
        max_infection_dur = min_infection_dur + 20 # 20 is a bit long to keep
        data['incubation_timer'][condition] = min_infection_dur 
        data['infection_timer'][condition] = np.random.randint(min_infection_dur, max_infection_dur, size=np.sum(condition))
        return data

    data = add_new_infections_np( data )
    return data

def migrate( data, timestep, num_infected=None ):
    # Migrate 1% of infecteds "downstream" every week; coz
    if timestep % settings.migration_interval == 0: # every week
        infected = np.where( data['infected'] )[0]
        fraction = int(len(infected)*0.05)
        selected = np.random.choice( infected, fraction )
        #print( f"Migrating {len(selected)} infecteds on day {timestep}." )
        # Update the 'nodes' array based on the specified conditions
        data['node'][selected] = np.where(data['node'][selected] == 0, settings.num_nodes - 1, data['node'][selected] - 1 )
    return data

def distribute_interventions( ctx, timestep ):
    def ria_9mo():
        condition_mask = (
            (ctx['age'] > 290/365.0) &
            (ctx['age'] < 291/365.0) &
            (ctx['immunity'] == 0) &
            (ctx['node'] == settings.campaign_node)
        )

        # Apply the update using the boolean mask
        ctx['immunity'][condition_mask] = 1
        ctx['immunity_timer'][condition_mask] = 3650
 
    def campaign( coverage = 1.0 ):
        # Create a boolean mask for the conditions specified in the WHERE clause
        condition_mask = (
            (ctx['immunity'] == 0) &
            (ctx['age'] < 16) &
            (ctx['node'] == settings.campaign_node)
        )

        # Shuffle the array to simulate ORDER BY RANDOM()
        #np.random.shuffle(ctx[condition_mask])

        # Get the indices of elements that satisfy the condition
        selected_indices = np.where(condition_mask)[0]

        # Calculate the number of elements to select based on the specified coverage
        num_to_select = int(len(selected_indices) * coverage)

        # Randomly select X% of indices
        selected_indices_subset = np.random.choice(selected_indices, size=num_to_select, replace=False)

        # Calculate the limit based on the specified coverage
        #limit = int(np.sum(condition_mask) * coverage)

        # Apply the update to the limited subset
        ctx['immunity'][selected_indices_subset] = 1
        ctx['immunity_timer'][selected_indices_subset] = -1

    ria_9mo()
    if timestep == settings.campaign_day:
        campaign(settings.campaign_coverage)
    return ctx

# Function to run the simulation for a given number of timesteps
def run_simulation(data, csvwriter, num_timesteps):
    currently_infectious, currently_sus, cur_reco  = collect_report( data )
    report.write_timestep_report( csvwriter, 0, currently_infectious, currently_sus, cur_reco )

    for timestep in range(1, num_timesteps + 1):
        data = update_ages( data )

        data = progress_infections( data )

        data = progress_immunities( data )

        new_infections = calculate_new_infections( data, currently_infectious, currently_sus )

        data = handle_transmission( data_in=data, new_infections_in=new_infections )

        data = add_new_infections( data )

        data = migrate( data, timestep )

        currently_infectious, currently_sus, cur_reco = collect_report( data )
        report.write_timestep_report( csvwriter, timestep, currently_infectious, currently_sus, cur_reco )


    print("Simulation completed. Report saved to 'simulation_report.csv'.")

# Main simulation
if __name__ == "__main__":
    data = initialize_database()

    # Create a CSV file for reporting
    csvfile = open('simulation_report.csv', 'w', newline='') 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Timestep', 'Node', 'Susceptible', 'Infected', 'Recovered'])

    # Run the simulation for 1000 timesteps
    run_simulation(data, csvwriter, num_timesteps=settings.duration )

