import subprocess
import sys
from pathlib import Path

import optuna
from objective import objective

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
storage_url = "sqlite:///optuna_study.db"  # SQLite file-based DB
sampler = optuna.samplers.CmaEsSampler()
study = optuna.create_study(
    sampler=sampler, direction="minimize", storage=storage_url, study_name="spatial_demo_calibration2", load_if_exists=True
)
study.optimize(objective, n_trials=1)

# Print the best parameters
print("Best parameters:", study.best_params)
# os.system("python3 impatient.py")
laser_script = Path("impatient.py").resolve(strict=True)
# subprocess.run(["python", "impatient.py"], check=True)
subprocess.run([sys.executable, str(laser_script)], check=True)
