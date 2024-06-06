#!/bin/bash

# Directory paths
#sandbox_dir="~/sandbox"
sandbox_dir="/var/tmp/sandbox"
src_dir="$(pwd)/src"

# Prompt the user to choose between England & Wales and CCS
PS3="Choose an option: "
options=("England & Wales" "CCS")
select choice in "${options[@]}"
do
    case $REPLY in
        1)
            england_wales="England & Wales"
            break
            ;;
        2)
            ccs="CCS"
            break
            ;;
        *)
            echo "Invalid option. Please choose 1 or 2."
            ;;
    esac
done

# Use the chosen value as needed
if [[ -n $england_wales ]]; then
    echo "You chose England & Wales."
elif [[ -n $ccs ]]; then
    echo "You chose CCS."
fi


# Create the sandbox directory if it doesn't exist
mkdir -p "$sandbox_dir"

pushd $sandbox_dir

mkdir model_numpy
mkdir model_sql

# Create symlinks for each file or directory in src_dir
cp "$src_dir/report.py" .
cp "$src_dir/../makefile" .
ln -sfn "$src_dir/../utils" .
cp "$src_dir/sir_numpy.py" .
cp "$src_dir/model_numpy/eula.py" model_numpy/
cp "$src_dir/sir_numpy_c.py" .
cp "$src_dir/update_ages.cpp" .
cp "$src_dir/measles.py" .
cp "$src_dir/../service/app.py" app.py
#ln -sfn "$src_dir/../service/requirements.txt" requirements.txt
cp "$src_dir/../service/requirements.txt" requirements.txt
pip3 install -r requirements.txt
#ln -sfn "$src_dir/births.csv"
#ln -sfn "$src_dir/cbrs.csv"

cp "$src_dir/../service/fits.npy" .
cp "$src_dir/settings.py" .
cp "$src_dir/post_proc.py" .

if [[ -n $england_wales ]]; then
    wget https://packages.idmod.org:443/artifactory/idm-data/laser/engwal_modeled.csv.gz
    wget https://packages.idmod.org:443/artifactory/idm-data/laser/attraction_probabilities.csv.gz
    wget https://packages.idmod.org:443/artifactory/idm-data/laser/cities.csv
    wget https://packages.idmod.org:443/artifactory/idm-data/laser/cbrs_ew.csv
    gunzip attraction_probabilities.csv.gz
    cp "$src_dir/demographics_settings_ew.py" ./demographics_settings.py
    cp "$src_dir/../Dockerfile_ew" ./Dockerfile
elif [[ -n $ccs ]]; then
    cp "$src_dir/sir_sql.py" .
    cp "$src_dir/model_sql/eula.py" model_sql/
    cp "$src_dir/demographics_settings_1node.py" ./demographics_settings.py
    cp "$src_dir/../Dockerfile_ccs" ./Dockerfile
    sed -i 's/migration_fraction=/migration_fraction=0#/g' settings.py
    make
fi

cp "$src_dir/../docker-compose.yml" .

make update_ages.so

echo "Symlinks & files created in $sandbox_dir"

