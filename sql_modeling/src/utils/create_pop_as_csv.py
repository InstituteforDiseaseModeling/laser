import sqlite3
import csv
import sys
sys.path.append( "." )
import settings
from sir_sql import initialize_database

# 1) Create a full population in a SQLite db in memory
conn = sqlite3.connect(":memory:")  # Use in-memory database for simplicity
initialize_database( conn, from_file=False )

# 2) Convert the modeled population into a csv file
print( f"Writing population file out out to csv: {settings.pop_file}." )
cursor = conn.cursor()
get_all_query = f"SELECT * FROM agents WHERE age<{settings.eula_age} ORDER BY age"

cursor.execute( get_all_query )
rows = cursor.fetchall()


with open( settings.pop_file, "w", newline='' ) as csvfile:
    csv_writer = csv.writer( csvfile )
    csv_writer.writerow( ['id', 'node', 'age', 'infected', 'infection_timer', 'incubation_timer', 'immunity', 'immunity_timer', 'expected_lifespan' ] )
    csv_writer.writerows( rows )

get_eula_query = f"SELECT node, CAST(age as INT) AS age, COUNT(*) AS total_individuals FROM agents WHERE age>={settings.eula_age} GROUP BY node, CAST(age as INT) ORDER BY node, age"
cursor.execute( get_eula_query )
rows = cursor.fetchall()

conn.close()

result_dict = [] # dict=array
# Iterate over the fetched rows and construct dictionaries
for row in rows:
    result_dict.append({
        "node": row[0],
        "age": row[1],
        "total": row[2]
    })

# Specify the CSV file path
csv_file_path = "eula_binned.csv"

# Write the dictionary data to a CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ["node", "age", "total"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for row in result_dict:
        writer.writerow(row)

