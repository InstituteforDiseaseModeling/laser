import polars as pl

# Example DataFrame
data = {'id': [1, 2, 3, 4, 5],
        'infected': [False, True, False, False, False],
        'immunity': [False, False, True, False, False]}

df = pl.DataFrame(data)

# Specify the conditions for filtering
condition = (pl.col("infected") == False) & (pl.col("immunity") == False)

# Subset of rows based on the conditions
subset_df = df.filter(condition)

import pdb
pdb.set_trace()
df.with_columns(
    subset_df.with_columns(
        pl.col("id")
        .is_in(["id"])
        .cast(int)
        .alias("id")
    )["id"]
)

# List of ids from the subset
ids_to_update = subset_df.select("id").to_pandas()["id"].tolist()

# Left outer join and filter rows where 'id' is not null from the right DataFrame
df = df.join(subset_df.select(["id"]), on="id", how="left").filter(pl.col("id").is_not_null())

# Update the 'infected' column in the filtered rows
df = df.with_columns(
    condition,
    "infected",
    pl.col("infected").apply(lambda x: 1 if x else 0)
)

# Display the updated DataFrame
print(df)

