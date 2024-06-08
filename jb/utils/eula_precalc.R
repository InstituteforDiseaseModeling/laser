library(data.table) # Ensure data.table is loaded

settings <- new.env()
source("demographics_settings.R", local = settings)

# Initialize the eula structure
init <- function() {
  eula <- list()
  data <- fread(settings$eula_file, skip = 1)
  
  for (i in 1:nrow(data)) {
    row <- data[i, ]
    node <- as.character(as.integer(row[[1]]))  # Convert node to character
    age <- as.integer(as.numeric(row[[2]]))
    total <- as.integer(row[[3]])
    
    if (!(node %in% names(eula))) {
      eula[[node]] <- list()
    }
    
    eula[[node]][[as.character(age)]] <- total  # Convert age to character
  }
  return(eula)
}

# Probability of dying array
makeham_parameter <- 0.008
gompertz_parameter <- 0.04
age_bins <- 0:101
probability_of_dying <- 2.74e-6 * (makeham_parameter + exp(gompertz_parameter * (age_bins - age_bins[1])))

#print("Probability of dying for each age:")
#print(probability_of_dying)

# Header and data writing functions
writeHeader <- function(file) {
  writeLines("t,node,pop", con = file)
}

writeData <- function(file, t, node, pop) {
  writeLines(paste(t, node, pop, sep = ","), con = file)
}

# Output file setup
args <- commandArgs(trailingOnly = TRUE)
output_file <- args[1]  # Make sure args[1] contains the output file path
output_file_conn <- file(output_file, "w")
on.exit(close(output_file_conn))

writeHeader(output_file_conn)

# Initialize eula
eula <- init()
#print("Initialized eula:")
#print(eula)

# Check structure of eula before the main loop
for (node in names(eula)) {
  #cat("Node:", node, "\n")
  #print(names(eula[[node]]))
}

# Calculate the expected deaths and new population for the next 20 years
for (t in 1:(20 * 365)) {
  for (node in names(eula)) {
    expected_deaths <- rep(0, length(eula[[node]]))
    #cat("Processing node:", node, "at time:", t, "\n")
    #print(names(eula[[node]])) # Print names of ages to verify structure
    
    for (age in names(eula[[node]])) {
      age_index <- as.integer(age)
      count <- eula[[node]][[age]]
      if (count > 0) {
        expected_deaths[age_index + 1] <- rbinom(1, count, probability_of_dying[age_index + 1])
      }
    }
    #cat("Expected deaths for node", node, "at time", t, ":", expected_deaths, "\n")
    
    for (age in names(eula[[node]])) {
      age_index <- as.integer(age)
      eula[[node]][[age]] <- eula[[node]][[age]] - expected_deaths[age_index + 1]
    }
    
    node_time_pop <- sum(unlist(eula[[node]]))
    #cat("Population for node", node, "at time", t, ":", node_time_pop, "\n")
    writeData(output_file_conn, t, node, node_time_pop)
  }
}

