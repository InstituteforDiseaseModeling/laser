library(httr)
library(jsonlite)

# Define the URL and the payload
url <- "http://172.18.0.2:5000/submit"
payload <- list(
  base_infectivity = 0.5,
  migration_fraction = 0.1,
  seasonal_multiplier = 1.2,
  duration = 2
)

# Convert the payload to JSON
json_payload <- toJSON(payload, auto_unbox = TRUE)

# Perform the POST request
response <- POST(
  url, 
  add_headers("Content-Type" = "application/json"), 
  body = json_payload, 
  encode = "json"
)

# Check the response
if (status_code(response) == 200) {
  print(content(response, "text"))
} else {
  print(paste("Request failed with status:", status_code(response)))
}

