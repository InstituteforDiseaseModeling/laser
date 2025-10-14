# Calibrate custom models

LASER can be calibrated using [Optuna](https://optuna.org/). Calibration is a process of tuning model parameters to fit real-world data, to ensure that model output provides accurate insights. Calibration can also be used as a method to help debug your model, as an inability to recreate known phenomena can help pinpoint issues in model code. For more information on calibration, check out IDM's [ModelingHub](https://institutefordiseasemodeling.github.io/modeling-hub/calibration/).


## Simple local calibration

1. Expose parameters in your model. Ensure your LASER model can load and apply parameters you wish to calibrate. These are typically passed through a `params` dictionary or a `PropertySet` and might include:

    - Basic reproduction number (R0)
    - Duration of infection
    - Seeding prevalence

1. Write post-processing code. Modify your model to save key outputs (e.g., number of infected individuals over time) to a CSV file. For example, use:

    `save_results_to_csv(sim.results)`

    This CSV will be used later by the objective function.

1. Create the objective function. Write a Python script, usually named `objective.py`, containing a function like this:

    ```
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
    ```

    Tip: You can write unit tests for your objective function by mocking model outputs.

1. Test the objective function standalone. Before integrating with Optuna, run your objective function directly to ensure it works:

    ```
    from objective import objective
    from optuna.trial import FixedTrial

    score = objective(FixedTrial({"R0": 2.5}))
    print(f"Test score: {score}")
    ```

    Expected result: A numeric score. If it crashes, check CSV paths and data types.

1. Run simple calibration (SQLite, no Docker). Use the calib/worker.py helper to run a local test study with a small number of trials.

    Linux/macOS (Bash or similar):

    ```
    export STORAGE_URL=sqlite:///example.db && python3 calib/worker.py --num-trials=10
    ```

    Windows (PowerShell):

    ```
    $env:STORAGE_URL="sqlite:///example.db"; python calib/worker.py --num-trials=10
    ```

    This is helpful for debugging. Consider running a scaled-down version of your model to save time.


## Local Dockerized calibration

1. Dockerize your model and objective. Use the provided `Dockerfile` to build a container that includes both your model and objective function. Do this from the main directory.

    `docker build . -f calib/Dockerfile -t idm-docker-staging.packages.idmod.org/laser/laser-polio:latest`

1. Create Docker network. You’ll need a shared network so your workers and database container can communicate:

    `docker network create optuna-network`

1. Launch MySQL database container:

    ```
    docker run -d --name optuna-mysql --network optuna-network -p 3306:3306 \
      -e MYSQL_ALLOW_EMPTY_PASSWORD=yes \
      -e MYSQL_DATABASE=optuna_db mysql:latest
    ```

1. Launch calibration worker:

    ```
    docker run --rm --name calib_worker --network optuna-network \
      -e STORAGE_URL="mysql://root@optuna-mysql:3306/optuna_db" \
      idm-docker-staging.packages.idmod.org/laser/laser-polio:latest \
      --study-name test_polio_calib --num-trials 1
    ```

    If that works, you can change the study name or number of trials.

    Troubleshooting: If this fails, try running the worker interactively and debug inside:

    `docker run -it --network optuna-network --entrypoint /bin/bash idm-docker-staging.packages.idmod.org/laser/laser-polio:latest`


1. Monitor calibration progress:

    Use Optuna CLI. You should be able to pip install optuna.

    ```
    optuna trials \
      --study-name=test_polio_calib \
      --storage "mysql+pymysql://root:@localhost:3306/optuna_db"

    optuna best-trial \
      --study-name=test_polio_calib \
      --storage "mysql+pymysql://root:@localhost:3306/optuna_db"
    ```


## Cloud calibration

1. Push Docker image to registry. If you’ve built a new docker image, you’ll want to push it so it’s available to AKS:

    `docker push idm-docker-staging.packages.idmod.org/laser/laser-polio:latest`

1. Cloud deployment. This step assumes you have secured access to an Azure Kubernetes Service (AKS) cluster. You may need to obtain or generate a new kube config file. Detailed instructions for that are not included here. This step assumes the cluster corresponding to your config is up and accessible.

    `cd calib/cloud`

    Edit `cloud_calib_config.py` to set the `storage_url` to:

    `"mysql+pymysql://optuna:superSecretPassword@localhost:3306/optunaDatabase"`

    Set the study name and number of trials per your preference. Detailed documentation of the other parameters is not included here.

    Launch multiple workers:

    `python3 run_calib_workers.py`

1. View final results:

    Forward port to local machine. Note that is the first instruction to rely on installing `kubectl`. Open a bash shell if necessary.

    `kubectl port-forward mysql-0 3306:3306 &`

    Use Optuna CLI to check results:

    ```
    optuna trials \
      --study-name=test_polio_calib \
      --storage "mysql+pymysql://optuna:superSecretPassword@localhost:3306/optunaDatabase"

    optuna best-trial \
      --study-name=test_polio_calib \
      --storage "mysql+pymysql://optuna:superSecretPassword@localhost:3306/optunaDatabase"
    ```

    Generate a report on disk about the study (can be run during study or at end).

    `python3 report_calib_aks.py`

    Launch Optuna dashboard:

    `python -c "import optuna_dashboard; optuna_dashboard.run_server('mysql+pymysql`


<!-- did not include the sections on workflow steps, iterative dev cycle, expected output, error handling, etc. as these feel like notes for an internal user. Not very relevant to public user base, but we can discuss :-) -->