import sqlite3
import random

# Connect to SQLite database (replace 'your_database.db' with your actual database file)
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# Create the 'agents' table
create_table_query = '''
    CREATE TABLE IF NOT EXISTS agents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        active INTEGER,
        countdown_timer INTEGER
    );
'''

cursor.execute(create_table_query)

# Populate the 'agents' table with 10,000 rows
for _ in range(10000):
    active_status = random.choice([0, 1])
    cursor.execute("INSERT INTO agents (active) VALUES (?)", (active_status,))

# Commit the changes for population
conn.commit()

# Define the update query to set countdown timer for active rows
update_query = '''
    UPDATE agents
    SET countdown_timer = ROUND(RANDOM() * (10 * 365 - 30) + 30)
    WHERE active = 1;
'''

# Execute the update query
cursor.execute(update_query)

# Commit the changes
conn.commit()

# Close the connection
conn.close()

