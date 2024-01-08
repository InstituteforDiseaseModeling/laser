# 1e6 SQL SEIRS Model

## Single node (just to get started)

![SIR plot](sql_seirs_output.png)

### Quick Intro
This model explores the idea and limits of creating a disease model entirely in SQL. Using Python & SQLite, to get started. 

Every row is an agent. Every column is an attribute. Every calculation is done as a SQL operation.

Note that plotting is a separate step.

SQL is SQLite for now. We then compare with Polars and NumPy. Potentially other SQL implementations.

Note that I started by calling this experiment "SQL Modeling" but some people fine SQL intimidating. It's probably better to call it "Dataframe Modeling". Because the whole model is essentially a single dataframe. SQL is just a natural choice for manipulating dataframes. But there are other popular SQL-wrappers nowadays like Pandas, etc. And in the limit, if you convert each of your dataframe columns to an numpy array, all the same things can be done with numpy. And we also open the door to numba implementations. But dataframes are very nice and easy to think about.

## 1e7 Numpy SEIRS Spatial Model

![Spatial_numpy_plot](10Magents_250nodes_migration.png)

### Summary
This model has 10 million agents, and 25 nodes. The population varies widely across the nodes, with population proportional to the node index. We seed the infection in node 25. We migrate 1% of infecteds "down" to the next node counting down. Here we plot just the top 25 nodes. We see infection spread from node to node. Incubation lasts 2 days. Infection last a week on average with some spread. Immunity lasts a month on average with some spread.

The simulation runs in a few minutes. I've done a 250 node one also which runs quickly if prevalence is low. Don't have any real eye-candy associated with that experiment yet.

This model was started in SQL but converted to numpy. 

99% of the coding (Python, SQLite, and numpy) was actually done with ChatGPT.

### What is a "SQL Model"?

Nothing more than a dataframe model. The entire model (at this point) is a single table or dataframe. Every agent is a row. Every attribute is a column. For example:

|id|node|age|infected|infection_timer|incubation_timer|immunity|immunity_timer|
|--|----|---|--------|---------------|----------------|--------|--------------|
|1|24|72.0|0|0|0|0|0|
|2|17|62.0|0|0|0|0|0|
|3|19|40.0|0|0|0|0|0|
|4|19|55.0|0|0|0|0|0|
|5|23|34.0|0|0|0|0|0|
|6|19|72.0|0|0|0|0|0|
|7|23|33.0|0|0|0|0|0|
|8|7|25.0|0|0|0|0|0|
|9|4|16.0|0|0|0|0|0|
|10|20|89.0|0|0|0|0|0|

Some advantages of this are:
- Dataframes are easy to intuit.
- The entire state of the model can be inspected using your preferred tabular data processing tools (SQL, Pandas, R).
- Serialization to/from csv files comes for free.
- Updating model state can be done via high performing tabular data manipulation tools (e.g., SQL, numpy, Pandas).
- It's almost trivial to go to and from SQL-based processing and numpy.
- In case it's not obvious from the above, the model can be run for some time, and the entire state written out to a csv. Then that csv can be analyzed and/or manipulated in a separate tool, and then the model can be resumed using the updated csv file as the input. There's no meaningful distinction between and input model dataframe and an output.

### Polars Exploration
I got the SQL model working in Polars. It was quite hard. Polars seems to be designed to do reads of your data very quickly, but updates are relatively hard to do (compared to SQL or numpy) and relatively slow. I am not a pandas person. It's possible that folks fluent in pandas would have found the Polars API more intuitive. I will get actual SQLite vs Polars perf numbers and add them here (done). There are some things I still haven't completely ported over to the Polars port (e.g., random  draws from distributions). Note that ChatGPT "support" for Polars is much weaker because Polars is pretty new and the latest API is newer than Chat3.5.

### Numpy Exploration
The numpy "port" was relatively easy even though I haven't used numpy before. All operations are essentially 1:1 comparable to the SQLite. While doing the numpy port I removed a for loop (over the nodes) and added parallelization to take advantage of available cores. It's not clear that this particular optimization can be done in the SQL version.

### Numba Exploration
I've found it very hard to get a working numba installation. I ended up using my personally owned Nvidia Jetson Nann and a docker image. They say that once you've removed all your foor loops, you've mostly taken away the thing that numba is best suited to optimize. The Non-cude numba version of the model was about 10x _slower_ than the numpy version. I haven't completed a numba.cuda port because there are much fewer numpy functions available for cuda. 

### Early Performance Numbers
|Model | Pop | Nodes | DB Indices | Reporting? | Other         | Time         | Observations|
|------|-----|-------|------------|------------|---------------|--------------|-------------|
|SQL   |1e6  | 25    | Node-id    | Off        |               | 14 minutes   | Too slow|
|SQL   |1e6  | 25    | Node-id    | On         |               | 13.5 minutes | Very odd|
|SQL   |1e6  | 25    | None       | On         |               | 13.5 minutes | Also odd|
|Numpy |1e6  | 25    | N/A        | On         | 12 cores      | 0.5 minutes  | Much faster than SQL|
|Numpy |1e6  | 250   | N/A        | On         | 12 cores      | 0.5 minutes  | 10x nodes didn't slow it down|
|Numpy |1e7  | 250   | N/A        | On         | 12 cores      | 3.0 minutes  | Prevalence was high|
|Polars|1e7  | 250   | None       | On         |               | 21.0 minutes | |

### Notes
- My disease transmission math doesn't normalize by node population size just yet.
- The SQL model build its population from a couple of parameters (pop size and num nodes). The other models start from an existing csv. I've just pushed a tool which creates a population csv from params. Instructions coming soon.

More soon...
