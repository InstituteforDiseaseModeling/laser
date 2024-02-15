#!/bin/bash

# Check if a filename and threshold are provided as arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <filename> <threshold>"
    exit 1
fi

# Assign the filename and threshold to variables
filename=$1
threshold=$2
# Save the header line to a variable
header=$(head -n 1 "$filename")

# Run awk to split the file based on the age column and the provided threshold
awk -v threshold="$threshold" -v header="$header" -F ',' '{
    if (NR == 1) {
        # Print the header to both output files
        print header > "eula_pop.csv"
        print header > "modeled_pop.csv"
    } else {
        if ($3 > threshold) {
            print > "eula_pop.csv"
        } else if ($3 < threshold) {
            print > "modeled_pop.csv"
        }
    }
}' "$filename"

