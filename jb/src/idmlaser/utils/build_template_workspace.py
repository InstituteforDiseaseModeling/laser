import os
import shutil
import subprocess
import pkg_resources

# Function to locate the examples directory in the idmlaser package
def get_idmlaser_examples_dir():
    try:
        import idmlaser
        package_path = pkg_resources.resource_filename('idmlaser', 'examples')
        return package_path
    except ImportError:
        print("idmlaser package is not installed.")
        exit(1)

# Get the examples directory path
src_dir = get_idmlaser_examples_dir()

# Prompt the user to choose a sandbox directory or use the default
default_sandbox_dir = "/var/tmp/sandbox"
sandbox_dir = input(f"Enter the sandbox directory path (default: {default_sandbox_dir}): ").strip()
if not sandbox_dir:
    sandbox_dir = default_sandbox_dir

# Check if the sandbox directory already exists
if os.path.exists(sandbox_dir):
    if not input(f"The directory '{sandbox_dir}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower() in {'y', 'yes'}:
        print("Exiting script. No changes were made.")
        exit()
    else:
        # Remove the directory and its contents
        shutil.rmtree(sandbox_dir)
        print(f"The directory '{sandbox_dir}' and its contents have been removed.")


# Prompt the user to choose between England & Wales and CCS
options = ["England & Wales", "CCS"]
choice = None
while choice not in [1, 2]:
    print("Choose an option:")
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    try:
        choice = int(input("Enter the number of your choice: "))
    except ValueError:
        choice = None
    if choice not in [1, 2]:
        print("Invalid option. Please choose 1 or 2.")

# Use the chosen value as needed
if choice == 1:
    chosen_option = "England & Wales"
    print("You chose England & Wales.")
else:
    chosen_option = "CCS"
    print("You chose CCS.")

# Create the sandbox directory if it doesn't exist
os.makedirs(sandbox_dir, exist_ok=True)

os.chdir(sandbox_dir)

# Copy makefile and settings.py
shutil.copy(os.path.join(src_dir, '../makefile'), '.')
shutil.copy(os.path.join(src_dir, 'settings.py'), '.')

# Download additional files and set up based on the chosen option
if chosen_option == "England & Wales":
    urls = [
        "https://packages.idmod.org:443/artifactory/idm-data/laser/engwal_modeled.csv.gz",
        "https://packages.idmod.org:443/artifactory/idm-data/laser/attraction_probabilities.csv.gz",
        "https://packages.idmod.org:443/artifactory/idm-data/laser/cities.csv",
        "https://packages.idmod.org:443/artifactory/idm-data/laser/cbrs_ew.csv",
        "https://packages.idmod.org:443/artifactory/idm-data/laser/fits_ew.npy"
    ]
    for url in urls:
        subprocess.run(['wget', url])
    
    subprocess.run(['gunzip', 'attraction_probabilities.csv.gz'])
    shutil.copy(os.path.join(src_dir, 'demographics_settings_ew.py'), './demographics_settings.py')
    shutil.copy(os.path.join('./fits_ew.npy'), './fits.npy')
else:
    shutil.copy(os.path.join(src_dir, 'demographics_settings_1node.py'), './demographics_settings.py')
    #subprocess.run(['cp', os.path.join(src_dir, '../Dockerfile_ccs'), './Dockerfile'])
    shutil.copy(os.path.join(src_dir, 'QuickStart.ipynb'), './QuickStart.ipynb')

    # Modify settings.py
    settings_path = 'settings.py'
    with open(settings_path, 'r') as file:
        settings_content = file.read()
    settings_content = settings_content.replace('migration_fraction=', 'migration_fraction=0#')
    with open(settings_path, 'w') as file:
        file.write(settings_content)

    subprocess.run(['make'])

#subprocess.run(['cp', os.path.join(src_dir, '../docker-compose.yml'), '.'])
#subprocess.run(['make', 'update_ages.so'])

print(f"Files and directories have been set up in {sandbox_dir}")

