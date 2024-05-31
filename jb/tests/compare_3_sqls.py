import sqlite3
import pandas as pd # as an assist
import polars as pl
from pyspark.sql import SparkSession
import time
import pdb
import settings

# Load data from the input file pop.csv into SQLite in-memory database
input_file = settings.pop_file

# SQLite with Pandas assist
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
df = pd.read_csv(input_file)
df.to_sql("agents", conn, if_exists='append', index=False)

# Polars
polars_df = pl.read_csv(input_file)

# Apache Spark SQL
spark = SparkSession.builder.appName("example").getOrCreate()
spark_df = spark.read.csv(input_file, header=True, inferSchema=True)

# Pandas
pd_df = df

# Query Execution
def execute_sqlite_query():
    start_time = time.time()
    cursor = conn.cursor()
    cursor.execute('''SELECT node, COUNT(*) FROM agents WHERE infected=0 AND immunity=0 GROUP BY node''')
    result = cursor.fetchall()
    cursor.close()
    return result, time.time() - start_time

def execute_polars_query():
    start_time = time.time()
    def polars_sql():
        ctx = pl.SQLContext(agents=polars_df, eager_execution=True)
        result = ctx.execute('SELECT node, COUNT(*) FROM agents WHERE infected=0 AND immunity=0 GROUP BY node')
        node_dict = dict(zip(result["node"].to_numpy(), result["node"]))
        count_dict = dict(zip(result["count"].to_numpy(), result["count"]))
        result_map = dict(zip(node_dict.values(), count_dict.values()))
        return result_map

    def polars_api():
        result = polars_df.filter((polars_df['infected'] == 0) & (polars_df['immunity'] == 0)).group_by('node').agg(pl.col('node').count().alias('count')).to_dict()
        node_dict = dict(zip(result["node"].to_numpy(), result["node"]))
        count_dict = dict(zip(result["count"].to_numpy(), result["count"]))
        result_map = dict(zip(node_dict.values(), count_dict.values()))
        return result_map

    return polars_sql(), time.time() - start_time

def execute_spark_query():
    start_time = time.time()
    result = spark_df.filter((spark_df['infected'] == 0) & (spark_df['immunity'] == 0)).groupBy('node').agg({'node': 'first', '*': 'count'}).toPandas().to_dict(orient='records')
    return result, time.time() - start_time

def execute_pandas_query():
    start_time = time.time()
    result_pandas = pd_df[(pd_df['infected'] == 0) & (pd_df['immunity'] == 0)].groupby('node').size().reset_index(name='count')

    # Convert the result to a list of dictionaries
    result = result_pandas.to_dict(orient='records')
    return result, time.time() - start_time

def test():
    # Execute queries
    result_sqlite, time_sqlite = execute_sqlite_query()
    result_polars, time_polars = execute_polars_query()
    result_spark, time_spark = execute_spark_query()
    result_pandas, time_pandas = execute_pandas_query()

    # Print results
    print("\n\nSQLite Result:", result_sqlite[150])
    print("SQLite Execution Time:", time_sqlite)

    print("\nPolars Result:", result_polars[150])
    print("Polars Execution Time:", time_polars)

    result_map = {item['node']: item['count(1)'] for item in result_spark}
    print("\nSpark Result:", result_map[150])
    print("Spark Execution Time:", time_spark)

    print("\nPandas Result:", result_pandas[150]['count'])
    print("Pandas Execution Time:", time_pandas)


for _ in range(5):
    test()
