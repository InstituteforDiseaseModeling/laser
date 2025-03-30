import os

import optuna
from objective import objective  # The objective function now runs inside the worker

# Connect to the same storage as the controller
# storage_url = get_storage_url()

# Load the study
# study = optuna.load_study(study_name="spatial_demo_calibr8n", storage=storage_url)

storage_url = os.getenv("STORAGE_URL", "mysql+pymysql://user:password@optuna-mysql/optuna_db")
study_name = "spatial_demo_calib_mar14"

try:
    study = optuna.load_study(study_name=study_name, storage=storage_url)
except KeyError:
    print(f"Study '{study_name}' not found. Creating a new study.")
    study = optuna.create_study(study_name=study_name, storage=storage_url)

# Run trials (each worker runs one or more trials)
study.optimize(objective, n_trials=5)  # Adjust per worker
