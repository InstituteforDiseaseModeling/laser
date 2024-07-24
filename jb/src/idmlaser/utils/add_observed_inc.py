import numpy as np
from scipy.stats import binom
import pandas as pd
import sys
import pdb

# Load the CSV file
input_file=sys.argv[1] if len(sys.argv)>1 else 'simulation_output.csv'
df = pd.read_csv(input_file)

# Set the parameters for the binomial distribution
num_trials = df['New_Infections']  # Number of trials (events)
prob_success = 0.5  # Probability of success (observation)

# Generate the binomial distribution
df['Observed Infections'] = binom.rvs(num_trials, prob_success)

# Sum "Observed Infections" over every 7 consecutive rows and create a new column "Week"
#df['Week'] = df.groupby(np.arange(len(df)) // 7)['Observed Infections'].transform('sum')
# Create a "Week" column by dividing the index by 7 (assuming the index starts from 0)
df['Week'] = df['Timestep'] // 7
#pdb.set_trace()
# Group by "Week" and "Node", and sum the "Observed Infections" over each group
df_grouped = df.groupby(['Week', 'Node']).agg({'Observed Infections': 'sum'}).reset_index()

# Select only the desired columns
#df_grouped = df[['Week', 'Node', 'Observed Infections']].sort_values( by=[ 'Week', 'Node' ] )
df_grouped = df_grouped.sort_values(by=['Week', 'Node'])

# Save the updated CSV file
#df.to_csv('simulation_output_with_obs.csv', index=False)
df_grouped.to_csv('simulation_output_with_obs_summed.csv', columns=['Week', 'Node', 'Observed Infections'], index=False)
