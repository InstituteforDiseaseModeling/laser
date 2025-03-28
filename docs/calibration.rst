Calibration Workflow for LASER Models
=====================================

This guide explains how to calibrate a LASER model using Optuna. Calibration is the process of adjusting model parameters (e.g., transmission rate, R0) so that simulation outputs match reference data (e.g., case counts, prevalence curves). This document assumes you've already built and tested a working LASER model.

Prerequisites
-------------
- A functioning, tested LASER model.
- Python environment with `laser-core`, `optuna`, `pandas`, and `numpy` installed.
- (Optional) Docker Desktop installed if running distributed calibration.

Simple Local Calibration
------------------------

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

Local Dockerized Calibration
----------------------------

6. **Dockerize Your Model and Objective**
   Use the provided `Dockerfile` to build a container that includes both your model and objective function. Do this from the main directory.

   .. code-block:: shell

       docker build . -f calib/Dockerfile -t idm-docker-staging.packages.idmod.org/laser/laser-polio:latest

7. **Create Docker Network**
   You'll need a shared network so your workers and database container can communicate:

   .. code-block:: shell

       docker network create optuna-network

8. **Launch MySQL Database Container**

   .. code-block:: shell

       docker run -d --name optuna-mysql --network optuna-network -p 3306:3306 \
         -e MYSQL_ALLOW_EMPTY_PASSWORD=yes \
         -e MYSQL_DATABASE=optuna_db mysql:latest

9. **Launch Calibration Worker**

    .. code-block:: shell

        docker run --rm --name calib_worker --network optuna-network \
          -e STORAGE_URL="mysql://root@optuna-mysql:3306/optuna_db" \
          idm-docker-staging.packages.idmod.org/laser/laser-polio:latest \
          --study-name test_polio_calib --num-trials 1

    If that works, you can change the study name or number of trials.

    **Troubleshooting:** If this fails, try running the worker interactively and debug inside:

    .. code-block:: shell

        docker run -it --network optuna-network --entrypoint /bin/bash idm-docker-staging.packages.idmod.org/laser/laser-polio:latest

10. **Monitor Calibration Progress**

    Use Optuna CLI. You should be able to pip install optuna.

    .. code-block:: shell

        optuna trials \
          --study-name=test_polio_calib \
          --storage "mysql+pymysql://root:@localhost:3306/optuna_db"

        optuna best-trial \
          --study-name=test_polio_calib \
          --storage "mysql+pymysql://root:@localhost:3306/optuna_db"

Cloud Calibration
------------------

11. **Push Docker Image to Registry**

    .. code-block:: shell

        docker push idm-docker-staging.packages.idmod.org/laser/laser-polio:latest

12. **Cloud Deployment**

    This step assumes you have secured access to an Azure Kubernetes Service cluster.

    - Create the study from Python:

      .. code-block:: shell

          cd calib
          python3 create_study.py

    - Launch multiple workers:

      .. code-block:: shell

          python3 run_calib_workers.py

13. **View Final Results**

    - Forward port to local machine. Note that is the first to rely on installing `kubectl`:

      .. code-block:: shell

          kubectl port-forward mysql-0 3306:3306 &

    - Use Optuna CLI to check results:

      .. code-block:: shell

          optuna trials \
            --study-name=test_polio_calib \
            --storage "mysql+pymysql://optuna:superSecretPassword@localhost:3306/optunaDatabase"

          optuna best-trial \
            --study-name=test_polio_calib \
            --storage "mysql+pymysql://optuna:superSecretPassword@localhost:3306/optunaDatabase"

Expected Output
---------------
- A best-fit parameter set (`R0`, etc.) that minimizes error.
- An Optuna study saved in MySQL or SQLite.
- Log files or CSVs showing score over time.

Error Handling
--------------
- Missing CSVs: Ensure output files are written by the model before scoring.
- Model crashes: Check Docker logs (`docker logs <container>`) or run interactively.
- Database connection errors: Confirm Docker network and container health. Ensure MySQL is listening on the expected port.

Next Steps
----------
Once you've completed calibration:
- Analyze the best-fit parameters.
- Re-run your model using the optimal settings.
- Generate plots or reports to summarize calibration quality.
