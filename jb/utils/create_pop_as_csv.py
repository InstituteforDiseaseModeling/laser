import sqlite3
import csv
import gzip
import sys
sys.path.append( "." )
import demographics_settings as settings
from sir_sql import initialize_database

# 1) Create a full population in a SQLite db in memory
print( f"Creating files to model population size {settings.pop} spread across {settings.num_nodes} nodes." )
conn = sqlite3.connect(":memory:")  # Use in-memory database for simplicity
initialize_database( conn, from_file=False )

# 2) Convert the modeled population into a csv file
print( f"Writing population file out out to csv: {settings.pop_file}." )
cursor = conn.cursor()
get_all_query = f"SELECT * FROM agents WHERE age<{settings.eula_age} ORDER BY age"

cursor.execute( get_all_query )
rows = cursor.fetchall()

print( f"Modeled population size = {len(rows)}" )

csv_output_file = settings.pop_file.strip( ".gz" )
with open( csv_output_file , "w", newline='' ) as csvfile:
    csv_writer = csv.writer( csvfile )
    csv_writer.writerow( ['id', 'node', 'age', 'infected', 'infection_timer', 'incubation_timer', 'immunity', 'immunity_timer', 'expected_lifespan' ] )
    csv_writer.writerows( rows )

print( f"Wrote uncompressed modeled population file as {csv_output_file}. Compressing..." )

# Open the input file in binary read mode
with open(csv_output_file, 'rb') as f_in:
    # Open the output file in binary write mode and compress it using gzip
    with gzip.open(settings.pop_file, 'wb') as f_out:
        # Read data from the input file and write it to the output file
        f_out.writelines(f_in)

print( "Compressed." )

get_eula_query = f"SELECT node, CAST(age as INT) AS age, COUNT(*) AS total_individuals FROM agents WHERE age>={settings.eula_age} GROUP BY node, CAST(age as INT) ORDER BY node, age"

cursor.execute( get_eula_query )
rows = cursor.fetchall()
conn.close()

result_dict = [] # dict=array
eula_pop = 0
# Iterate over the fetched rows and construct dictionaries
for row in rows:
    result_dict.append({
        "node": row[0],
        "age": row[1],
        "total": row[2]
    })
    eula_pop += row[2]
print( f"EULA population size = {eula_pop}" )

# Specify the CSV file path
csv_file_path = settings.eula_file

# Write the dictionary data to a CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ["node", "age", "total"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for row in result_dict:
        writer.writerow(row)

