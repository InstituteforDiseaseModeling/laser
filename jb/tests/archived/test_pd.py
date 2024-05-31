import pandas as pd

# Example DataFrame
data = {'id': [1, 2, 3, 4, 5],
        'infected': [False, True, False, False, False],
        'immunity': [False, False, True, False, False]}

df = pd.DataFrame(data)

# Specify the conditions for filtering
condition = (df['infected'] == False) & (df['immunity'] == False)

# Subset of rows based on the conditions
subset_df = df[condition]

# List of ids from the subset
ids_to_update = subset_df['id'].tolist()

# Update the 'infected' column in the main DataFrame for selected ids
df.loc[df['id'].isin(ids_to_update), 'infected'] = True

# Display the updated DataFrame
print(df)

