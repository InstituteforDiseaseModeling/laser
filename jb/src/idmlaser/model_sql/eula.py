import sqlite3

eula_cursor = None
def init():
    eula = sqlite3.connect( "eula.db" )
    global eula_cursor
    eula_cursor = eula.cursor()

def get_recovereds_by_node():
    eula_cursor.execute('SELECT node, COUNT(*) FROM agents GROUP BY node')
    recovered_counts_db = eula_cursor.fetchall()
    return dict(recovered_counts_db)

def update_natural_mortality():
    eula_cursor.execute( "DELETE FROM agents WHERE age>=expected_lifespan" )
