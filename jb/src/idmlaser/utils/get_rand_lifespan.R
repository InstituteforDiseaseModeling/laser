library(stats)

settings <- new.env()
source("demographics_settings.R", local = settings)
#pop_below_eula <- as.integer(0.05*settings$pop)

scaled_samples <- NULL
lifespan_idx <- 0

get_beta_samples <- function(number) {
  # Define parameters
  lifespan_mean <- 75
  lifespan_max_value <- 110

  # Scale and shift parameters to fit beta distribution
  alpha <- 4  # Adjust this parameter to increase lower spread
  beta_ <- 2
  samples <- rbeta(number, alpha, beta_)
  scaled_samples <- samples * (lifespan_max_value - 1) + 1
  return(scaled_samples)
}

get_rand_lifespan <- function() {
  beta_lifespan <- function() {
    # Generate random samples from the beta distribution
    if (is.null(scaled_samples)) {
      scaled_samples <<- get_beta_samples(settings$pop)
    }

    # Scale samples to match the desired range
    ret_value <- scaled_samples[lifespan_idx]
    lifespan_idx <<- lifespan_idx + 1
    return(ret_value)
  }

  return(beta_lifespan())
}

