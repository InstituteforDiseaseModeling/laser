Calibration Workflow for LASER Models
=====================================

This guide explains how to calibrate a LASER model using Optuna. Calibration is the process of adjusting model parameters (e.g., transmission rate, R0) so that simulation outputs match reference data (e.g., case counts, prevalence curves). This document assumes you've already built and tested a working LASER model.

Prerequisites
-------------
- A functioning, tested LASER model.
- Python environment with `laser-core`, `optuna`, `pandas`, and `numpy` installed.
- (Optional) Docker Desktop installed if running distributed calibration.

Steps
-----

1. **Expose Parameters in Your Model**
   Ensure your LASER model can load and apply parameters you wish to calibrate. These are typically passed through a `params` dictionary or a `PropertySet` and might include:

   - Basic reproduction number (R0)
   - Duration of infection
   - Seeding prevalence

2. **Write Post-Processing Code**
   Modify your model to save key outputs (e.g., number of infected individuals over time) to a CSV file. For example, use:

   .. code-block:: python

       save_results_to_csv(sim.results)

   This CSV will be used later by the objective function.

3. **Create the Objective Function**
   Write a Python script, usually named `objective.py`, containing a function like this:

   .. code-block:: python

       def objective(trial):
           # Load trial parameters
           R0 = trial.suggest_float("R0", 1.0, 3.5)

           # Run model (via subprocess, or function call)
           run_model(R0)

           # Load model output and reference data
           model_df = pd.read_csv("output.csv")
           ref_df = pd.read_csv("reference.csv")

           # Compare and return score
           error = np.mean((model_df["I"] - ref_df["I"])**2)
           return error

   **Tip:** You can write unit tests for your objective function by mocking model outputs.

4. **Test Objective Function Standalone**
   Before integrating with Optuna, run your objective function directly to ensure it works:

   .. code-block:: python

       from objective import objective
       from optuna.trial import FixedTrial

       score = objective(FixedTrial({"R0": 2.5}))
       print(f"Test score: {score}")

   **Expected Result:** A numeric score. If it crashes, check CSV paths and data types.

5. **Run Simple Calibration (SQLite, No Docker)**
   Use the `run_optuna.py` helper to run a local test study with a small number of trials:

   .. code-block:: shell

       python run_optuna.py --study-name test --num-trials 5 --storage sqlite:///example.db

   This is helpful for debugging. Consider running a scaled-down version of your model to save time.

6. **Dockerize Your Model and Objective**
   Use the provided `Dockerfile` to build a container that includes both your model and objective function.

   .. code-block:: shell

       docker build -t calib-worker .

7. **Create Docker Network**
   You'll need a shared network so your workers and database container can communicate:

   .. code-block:: shell

       docker network create optuna-network

8. **Set Environment Variables**
   Define these in your shell:

   .. code-block:: shell

       export MYSQL_USER=optuna
       export MYSQL_PASSWORD=superSecretPassword
       export MYSQL_DB=optunaDatabase
       export MYSQL_ROOT_PASSWORD=root-password
       export NUM_TRIALS=100
       export STUDY_NAME=test_polio_calib

9. **Launch MySQL Database Container**

   .. code-block:: shell

       docker run --rm --network optuna-network \
         --name optuna-mysql \
         -e MYSQL_ROOT_PASSWORD=$MYSQL_ROOT_PASSWORD \
         -e MYSQL_DATABASE=$MYSQL_DB \
         -e MYSQL_USER=$MYSQL_USER \
         -e MYSQL_PASSWORD=$MYSQL_PASSWORD \
         -d mysql:8

10. **Launch Calibration Worker**

    .. code-block:: shell

        docker run --rm --network optuna-network \
          -e STORAGE_URL="mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@optuna-mysql:3306/${MYSQL_DB}" \
          calib-worker:latest

    **Troubleshooting:** If this fails, try running the worker interactively and debug inside:

    .. code-block:: shell

        docker run -it --network optuna-network --entrypoint /bin/bash calib-worker:latest

11. **Monitor Calibration Progress**

    Use Optuna CLI:

    .. code-block:: shell

        optuna trials \
          --study-name=$STUDY_NAME \
          --storage "mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@localhost:3306/${MYSQL_DB}"

        optuna best-trial \
          --study-name=$STUDY_NAME \
          --storage "mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@localhost:3306/${MYSQL_DB}"

12. **Push Docker Image to Registry (Optional)**

    .. code-block:: shell

        docker tag calib-worker:latest your-registry/laser/laser-polio:latest
        docker push your-registry/laser/laser-polio:latest

13. **Cloud Deployment (Optional)**

    If running in the cloud (e.g., Azure):

    - Create the study from Python:

      .. code-block:: shell

          python3 run_create_study.py

    - Forward port to local machine:

      .. code-block:: shell

          kubectl port-forward mysql-0 3306:3306 &

    - Launch multiple workers:

      .. code-block:: shell

          python3 run_workers.py

14. **View Final Results**

    Same Optuna CLI commands as before. You can also export all trials for visualization or further analysis.

Expected Output
---------------
- A best-fit parameter set (`R0`, etc.) that minimizes error.
- An Optuna study saved in MySQL or SQLite.
- Log files or CSVs showing score over time.

Error Handling
--------------
- Missing CSVs: Ensure output files are written by the model before scoring.
- Model crashes: Check Docker logs (`docker logs <container>`) or run interactively.
- Database connection errors: Confirm network and env vars. MySQL must be reachable from workers.

Next Steps
----------
Once you've completed calibration:
- Analyze the best-fit parameters.
- Re-run your model using the optimal settings.
- Generate plots or reports to summarize calibration quality.
