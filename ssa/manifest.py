import importlib.resources as pkg_resources
import importlib
import os

psi_data="pred_psi_processed.csv"
#psi_data="psi_synth_allsame.csv"
#age_data=pkg_resources.path('idmlaser_cholera', 'USA-pyramid-2023.csv') # meh
age_data="nigeria_pyramid.csv"
seasonal_dynamics = "pred_seasonal_dynamics_processed.csv"
immune_decay_params = "immune_decay_params.json"
laser_cache = "laser_cache"
#wash_theta="param_theta_WASH.csv"
#population_file = "nigeria.py"
#population_file = "nigeria_onenode.py"
#population_file = "synth_10_allsame.py"
#population_file = "synth_25.py"
population_file = "synth_small_ssa.py"



def load_population_data():
    if population_file is None:
        raise ValueError("A data file path must be specified.")

    # Check if the data file is a Python module
    if os.path.isfile(population_file) and population_file.endswith('.py'):
        spec = importlib.util.spec_from_file_location("location_data", population_file)
        location_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(location_data)

        return location_data.run()
    else:
        raise ValueError(f"Invalid data file '{population_file}'. It must be a Python file.")

