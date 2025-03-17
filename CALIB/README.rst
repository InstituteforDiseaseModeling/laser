LASER Calibration with Optuna
=============================

This repository demonstrates how to calibrate a spatial **Susceptible-Infected-Recovered (SIR) model** using **Optuna**, an optimization framework for hyperparameter tuning.

Our goal is to identify **transmission rate** and **migration rate** values that produce a **second prevalence peak** in **node 0**, with a noticeable **trough between the peaks**.

Model Overview
--------------

The core model (``laser.py``) implements a spatial **SIR model** with the following characteristics:

- **500,000 agents** spread **heterogeneously** across **20 nodes**.
- **Node 0** is the **largest**, while **node 19** is the **smallest**.
- The outbreak is **seeded in 1%** of the **population of node 0**.
- **Gravity-based migration** determines agent movement between nodes.
- Infections last **5 to 15 days** (uniformly distributed).
- **Configurable** transmission and migration rates.

Calibration Goal
----------------

We use **Optuna** to optimize the **transmission rate** and **migration rate** to achieve:

✅ A **second prevalence peak** in node 0.
✅ A **clear trough** between the two peaks.

The calibration process runs multiple simulations, adjusting parameters until the desired epidemic curve is achieved.

Files and Structure
-------------------

- ``laser.py`` – Implements the SIR model.
- ``run.py`` – Main calibration script (starts the Optuna process).
- ``objective.py`` – Defines the **objective function**, evaluating how well each trial matches the target epidemic curve.
- ``impatient.py`` – Allows you to inspect the **current best parameters** while calibration is still running.
- ``optuna_review_contour.py`` – Generates **Optuna visualizations** to analyze the search process.

Installation
------------

Before running the calibration, install the required dependencies:

.. code-block:: bash

    pip install optuna numpy matplotlib

Running Calibration
-------------------

To start calibration, run:

.. code-block:: bash

    python run.py

By default, this will:

- Run **100 trials** with **4 replicates** per trial.
- Simulate **500 timesteps per run** (each taking ~10 seconds).
- Identify the best parameter set and run **a final simulation** with those values.

Checking Calibration Progress
-----------------------------

To monitor the best parameters found so far, run:

.. code-block:: bash

    python impatient.py

To visualize the parameter search space explored by Optuna, run:

.. code-block:: bash

    python optuna_review_contour.py

Expected Results
----------------

If calibration is successful, the final prevalence plot for **node 0** should display:

✅ A **clear second peak** in infections.
✅ A **noticeable trough** between the two peaks.

You can modify parameters in the scripts to explore different calibration behaviors.

Next Steps
----------

- Try adjusting the search space or evaluation criteria in ``objective.py``.
- Increase the number of trials to improve calibration accuracy.
- Experiment with different outbreak seeding strategies.

Summary
-------

This project demonstrates **LASER and Optuna-based epidemic model calibration** and is designed for researchers interested in large scale spatial disease modeling and parameter estimation.

Dockerized
----------

Network Start
^^^^^^^^^^^^^

docker network create optuna-network

DB Start
^^^^^^^^

docker run -d --name optuna-mysql --network optuna-network -p 3306:3306 -e MYSQL_ALLOW_EMPTY_PASSWORD=yes -e MYSQL_DATABASE=optuna_db mysql:latest

Optuna Workers
^^^^^^^^^^^^^^

docker build -t docker.io/library/calib-worker:latest
docker run --rm --name calib-worker --network optuna-network -e STORAGE_URL="mysql+pymysql://root@optuna-mysql/mysql" docker.io/library/calib-worker:latest&

View Results
^^^^^^^^^^^^

python3 impatient_mysqldocker.py
