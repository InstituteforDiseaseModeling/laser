import json
import subprocess
import sys
from pathlib import Path

import optuna

# Define the storage URL
storage_url = "sqlite:///optuna_study.db"
study = optuna.create_study(direction="minimize", storage=storage_url, study_name="spatial_demo_calibration2", load_if_exists=True)

# Get the best parameters found so far
best_params = study.best_params
print("Best parameters so far:", best_params)

# Add fixed parameters
best_params.update({"population_size": 500000, "nodes": 20, "timesteps": 300, "initial_infected_fraction": 0.01})

# Save best parameters
Path("params_test.json").write_text(json.dumps(best_params, indent=4))
print("Saved best parameters to params_test.json.")

# Run the model
subprocess.run([sys.executable, "laser_run.py"], check=True)
