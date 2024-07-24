library(data.table)
library(nlstools)

settings <- new.env()
source("demographics_settings.R", local = settings)

# Function to perform linear fit
linear_fit <- function(x, m, b) {
  return(m * x + b)
}

# Read the CSV file
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- settings$eula_pop_fits

df <- fread(input_file)

# Group data by 'node' and fit a line for each group
fits <- list()
nodes <- unique(df$node)

for (node in nodes) {
  group <- df[node == df$node, ]
  x_data <- group$t
  y_data <- group$pop
  
  # Perform linear regression
  model <- nls(y_data ~ linear_fit(x_data, m, b), start = list(m = 1, b = 0))
  params <- coef(model)
  
  # Save the fit parameters to a list
  fits[[as.character(node)]] <- params
}

# Save the fits list to an RDS file
saveRDS(fits, output_file)

cat("Fits saved to", output_file, "\n")

