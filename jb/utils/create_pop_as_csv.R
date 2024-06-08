library(RSQLite)
library(dplyr)
library(data.table)
library(DBI)

# Load settings from an R file similar to 'demographics_settings'
# library(demographics_settings)
settings <- new.env()
source("demographics_settings.R", local = settings)

# List all objects in the settings environment
setting_names <- ls(settings)

# Print each value
for (name in setting_names) {
  cat(name, "=", get(name, envir = settings), "\n")
}

source( "utils/get_rand_lifespan.R" )
get_rand_lifespan() # weird issue where it needs to be called once to init
expected_lifespans <- numeric(settings$pop)

# Populate the expected lifespans vector using get_rand_lifespan() for each row
for (i in 1:settings$pop) {
  expected_lifespans[i] <- get_rand_lifespan()
}

get_node_ids <- function() {
  # Generate the array based on the specified conditions
  node_list <- lapply(settings$nodes, function(node) {
    rep(node, node + 1)
  })

  array <- unlist(node_list)

  # Repeat the array to match the population size
  repeats <- ceiling(settings$pop / length(array))
  array <- rep(array, times = repeats)[1:settings$pop]

  # Shuffle the array to randomize the order
  array <- sample(array)

  # Convert the array to integers
  array <- as.integer(array)

  # Print the first few elements as an example
  # print(head(array, 20))

  return(array)
}

initialize_database <- function(conn) {
  print("Initializing pop NOT from file.")
  
  # Check if connection is provided
  if (is.null(conn)) {
    # Use in-memory database for simplicity
    conn <- dbConnect(SQLite(), ":memory:")
  }
  
  # Create agents table
  dbExecute(conn, "
             CREATE TABLE agents (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               node INTEGER,
               age REAL,
               infected BOOLEAN,
               infection_timer INTEGER,
               incubation_timer INTEGER,
               immunity BOOLEAN,
               immunity_timer INTEGER,
               expected_lifespan INTEGER
             )
             ")
  
  # Create indices
  dbExecute(conn, "CREATE INDEX immunity_idx ON agents(immunity)")
  dbExecute(conn, "CREATE INDEX immunity_node_idx ON agents(immunity, node)")
  
  # Insert agents data
  
  print( "Initializing agents_data now..." )
  individual_modeled_data <- data.frame(
    node = get_node_ids(), # sample(0:100, settings$pop, replace = TRUE), # rep(node_assignments, each = settings$pop),
    age = runif(settings$pop, min = 0, max = 100) + runif(settings$pop, min = 0, max = 365) / 365,
    infected = rep(FALSE, settings$pop),
    infection_timer = rep(0, settings$pop),
    incubation_timer = rep(0, settings$pop),
    immunity = rep(FALSE, settings$pop),
    immunity_timer = rep(0, settings$pop),
    expected_lifespan = expected_lifespans
  )
  agents_data <- individual_modeled_data 
  dbWriteTable(conn, "agents", agents_data, append = TRUE)
 
  # Seed exactly 100 people to be infected in the first timestep
  #big_node <- settings$num_nodes - 1
  #infected_agents <- sample(1:settings$pop, 100, replace = FALSE)
  # dbExecute(conn, "
  #           UPDATE agents 
  #           SET infected = 1, infection_timer = 9 + CAST(RANDOM() * 6 AS INT), incubation_timer = 3 
  #           WHERE node = ? AND id IN (?)
  #           ", big_node, infected_agents)
  
  return(conn)
}


# 1) Create a full population in a SQLite db in memory
cat(sprintf("Creating files to model population size %d spread across %d nodes.\n", settings$pop, settings$num_nodes))
conn <- dbConnect(SQLite(), ":memory:")
initialize_database(conn)
dbListTables(conn)  # Check if the 'agents' table exists

# 2) Convert the modeled population into a csv file
cat(sprintf("Writing population file out to csv: %s.\n", settings$pop_file))

get_all_query <- sprintf("SELECT * FROM agents WHERE age < %d ORDER BY age", settings$eula_age)

# Print the query for debugging
#print(paste("SQL Query:", get_all_query))

# Validate the query string
if (!is.character(get_all_query) || length(get_all_query) != 1) {
  stop("get_all_query must be a single string")
}

# Print the query for debugging
print(paste("SQL Query:", get_all_query))

# Fetch data from the database
print("Fetching data from the database...")
agents_data <- tryCatch({
  dbGetQuery(conn, get_all_query)
}, error = function(e) {
  print("Error during dbGetQuery:")
  print(e)
  NULL
})

# Check if data was fetched successfully
if (is.null(agents_data)) {
  stop("Failed to fetch data from the database.")
}

agents_data <- dbGetQuery(conn, get_all_query)
#print(agents_data)

cat(sprintf("Modeled population size = %d\n", nrow(agents_data)))

csv_output_file <- gsub(".gz$", "", settings$pop_file)
fwrite(agents_data, csv_output_file)

cat(sprintf("Wrote uncompressed modeled population file as %s. Compressing...\n", csv_output_file))

# Compress the CSV file
gzfile_conn <- gzfile(settings$pop_file, "wb")
writeBin(readBin(csv_output_file, "raw", file.info(csv_output_file)$size), gzfile_conn)
close(gzfile_conn)

cat("Compressed.\n")

get_eula_query <- sprintf("SELECT node, CAST(age AS INT) AS age, COUNT(*) AS total_individuals FROM agents WHERE age >= %d GROUP BY node, CAST(age AS INT) ORDER BY node, age", settings$eula_age)
eula_data <- dbGetQuery(conn, get_eula_query)
dbDisconnect(conn)

eula_pop <- sum(eula_data$total_individuals)
cat(sprintf("EULA population size = %d\n", eula_pop))

# Write the dictionary data to a CSV file
csv_file_path <- settings$eula_file
fwrite(eula_data, csv_file_path)

