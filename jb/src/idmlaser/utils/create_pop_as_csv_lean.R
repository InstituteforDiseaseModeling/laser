library(RSQLite)
library(dplyr)
library(data.table)

settings <- new.env()
source("demographics_settings.R", local = settings)
pop_below_eula <- as.integer(0.05*settings$pop)

# Format the value of settings$pop_below_eula before printing
# pop_below_eula_formatted <- as.character(settings$pop_below_eula)

# Print the formatted value of settings$pop_below_eula
pop_below_eula # print

source( "utils/get_rand_lifespan.R" )
get_rand_lifespan()

#node_assignments <- sample(0:(settings$num_nodes - 1), settings$pop, replace = TRUE)

expected_lifespans <- numeric(pop_below_eula)

# Populate the expected lifespans vector using get_rand_lifespan() for each row
for (i in 1:pop_below_eula) {
  expected_lifespans[i] <- get_rand_lifespan()
}

get_node_ids <- function() {
  # Generate the array based on the specified conditions
  node_list <- lapply(settings$nodes, function(node) {
    rep(node, node + 1)
  })

  array <- unlist(node_list)

  # Repeat the array to match the population size
  repeats <- ceiling(pop_below_eula / length(array))
  array <- rep(array, times = repeats)[1:pop_below_eula]

  # Shuffle the array to randomize the order
  array <- sample(array)

  # Convert the array to integers
  array <- as.integer(array)

  # Print the first few elements as an example
  # print(head(array, 20))

  return(array)
}

individual_data <- data.frame(
  node = get_node_ids(), # sample(0:100, pop_below_eula, replace = TRUE), # rep(node_assignments, each = pop_below_eula),
  age = runif(pop_below_eula, min = 0, max = settings$eula_age) + runif(pop_below_eula, min = 0, max = 365) / 365, # runif(pop_below_eula, min = 0, max = 100), # 
  infected = rep(FALSE, pop_below_eula),
  infection_timer = rep(0, pop_below_eula),
  incubation_timer = rep(0, pop_below_eula),
  immunity = rep(FALSE, pop_below_eula),
  immunity_timer = rep(0, pop_below_eula),
  expected_lifespan = expected_lifespans
)
