# Population and Nodes if building population from parameters
pop <- as.integer(2.4e6) + 1
num_nodes <- 2
nodes <- seq(0, num_nodes - 1)

# Epidemiologically Useless Light Agents (EULA) Age
eula_age <- 5

# Filenames if loading population from file
pop_file <- "modeled_pop.csv.gz"
eula_file <- "eula_binned.csv"
eula_pop_fits <- "fits.npy"
cbr_file <- "cbrs.csv"

