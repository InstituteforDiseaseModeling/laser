import subprocess
import sys
from pathlib import Path

import optuna
from objective import objective

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
storage_url = "sqlite:///optuna_study.db"  # SQLite file-based DB
# You can use non-default samplers if you want; we'll go with the default
# sampler = optuna.samplers.CmaEsSampler()
study = optuna.create_study(
    # sampler=sampler,
    direction="minimize",
    storage=storage_url,
    study_name="spatial_demo_calibration",
    load_if_exists=True,
)
study.optimize(objective, n_trials=1)  # n_trials is how many more trials; it will add to an existing study if it finds it in the db.

# Print the best parameters
print("Best parameters:", study.best_params)

laser_script = Path("impatient.py").resolve(strict=True)
subprocess.run([sys.executable, str(laser_script)], check=True)
