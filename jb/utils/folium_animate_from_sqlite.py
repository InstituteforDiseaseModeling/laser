import csv
import sqlite3
import random
import numpy as np
import requests
import os
import argparse
import os

import folium
from folium.plugins import HeatMapWithTime

# Connect to the SQLite database (or create it)
conn = sqlite3.connect(':memory:')
#conn = sqlite3.connect('ew.db')
cursor = conn.cursor()

def preproc( sim_report_file ):
    # Create tables and import data from CSV files
    cursor.execute("DROP TABLE IF EXISTS engwal")
    cursor.execute("DROP TABLE IF EXISTS cities")

    # Import simulation_output.csv into engwal table
    with open(sim_report_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        cursor.execute(f"CREATE TABLE engwal ({', '.join(headers)})")
        cursor.executemany(f"INSERT INTO engwal VALUES ({', '.join(['?' for _ in headers])})", reader)

    # Import cities.csv into cities table
    if os.path.exists( 'cities.csv' ):
        with open('cities.csv', 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            cursor.execute(f"CREATE TABLE cities ({', '.join(headers)})")
            cursor.executemany(f"INSERT INTO cities VALUES ({', '.join(['?' for _ in headers])})", reader)
    else:
        url = 'https://packages.idmod.org:443/artifactory/idm-data/laser/cities.csv'

        # Perform an HTTP GET request
        response = requests.get(url)
        response.raise_for_status()  # Check that the request was successful

        # Process the CSV file directly from the response content
        content = response.content.decode('utf-8').splitlines()
        reader = csv.reader(content)
        headers = next(reader)
        cursor.execute(f"CREATE TABLE cities ({', '.join(headers)})")
        cursor.executemany(f"INSERT INTO cities VALUES ({', '.join(['?' for _ in headers])})", reader)

    
    # Create the view
    cursor.execute("""
    CREATE VIEW cases AS
    SELECT Timestep, Node, Name, Latitude, Longitude, New_Infections
    FROM engwal, cities
    WHERE engwal.Node = cities.ID
    """)

    conn.commit()


def process( output_file ):
    # Create a map centered at a specific location
    birmingham_location = (52.485,-1.86)
    m = folium.Map(location=(birmingham_location[0],birmingham_location[1]), zoom_start=8) # Create a list to store the data for HeatMapWithTime

    start_time=800
    cursor.execute(f'SELECT CAST(Timestep AS INT), Latitude, Longitude, CAST(New_Infections AS INT) FROM cases WHERE (CAST(Timestep AS INT)>{start_time})')
    raw_data = cursor.fetchall()

    # Group the data by timestep
    grouped_data = {}
    for row in raw_data:
        timestep, lat, lon, new_infections = row
        if timestep not in grouped_data:
            grouped_data[timestep] = []
        grouped_data[timestep].append([lat, lon, new_infections])

    data = []
    grouped_data = list(grouped_data.values())
    locations=[[ float(elem[0]), float(elem[1]) ] for elem in grouped_data[0]]

    max_cases = 500 # updating this manually
    for t in range(len(grouped_data)-1):
        this_row = []
        for i, location in enumerate( locations ):
            try:
                def renorm( value ):
                    if value > max_cases:
                        value = max_cases
                    offset = 1
                    log_value = np.log(value + offset)
                    log_min = np.log(0 + offset)
                    log_max = np.log(max_cases + offset)
                    return (log_value - log_min) / (log_max - log_min)
                case_value = grouped_data[t][i][2]
                case_value = renorm( case_value )
                # random jitter seems to be essential!!! Yes, wack.
                this_row.append( [ location[0], location[1], case_value+random.random()/2000 ] )
            except Exception as ex:
                print( f"Exception with t={t}, i={i}." )
                raise ValueError( str( ex ) )
        data.append( this_row )

    heat_map = HeatMapWithTime(
            data,
            radius=15, 
            gradient={0.0: 'blue', 0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 0.8: 'red'},
            min_opacity = 0.0,
            max_opacity = 0.8,
            #use_local_extrema=True,
            auto_play=True 
        )
    heat_map.add_to(m)

    # Save the map as an HTML file
    if not os.path.exists( "html" ):
        os.mkdir( "html" )
    m.save( output_file )

    print( f"File written: {output_file}" )

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process simulation data.')
    parser.add_argument('-i', '--input', type=str, default='simulation_output.csv',
                        help='Input file name (default: simulation_output.csv)')
    parser.add_argument('-o', '--output', type=str, default='html/sim_animation.html',
                        help='Output file name (default: sim_animation.html)')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the preproc function with the input file
    preproc(args.input)
    
    # Call the process function with the output file
    process(args.output)

    print( "Either load the html file directory or serve it by entering" )
    print( "python3 -m http.server --directory html 4444" )
    print( "And entering 'http://localhost:4444/sim_animation.html' in a browser." )
