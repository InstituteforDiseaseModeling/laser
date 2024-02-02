import sqlite3 as sql
import csv
import pandas as pd
import sys

df = pd.read_csv( sys.argv[1] )
df = df.drop(columns=['mcw'])
conn=sql.connect("eula.db")
df.to_sql("agents",conn,index=False)

