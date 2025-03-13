import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

laser_script = Path("laser.py").resolve(strict=True)


def evaluate_3_weights_better(results_csv):
    """Evaluate if the I series of Node 0 has a second peak, while smoothly guiding towards it."""

    # Load simulation results
    df = pd.read_csv(results_csv)

    # Filter for Node 0
    node_0_data = df[df["Node"] == 0]

    # Extract infected column and total population
    infected_series = node_0_data["Infected"].values
    total_population = (node_0_data["Susceptible"] + node_0_data["Infected"] + node_0_data["Recovered"]).values

    # Normalize infected series to prevalence
    infected_prevalence = infected_series / total_population

    # Find peaks
    peaks, _ = find_peaks(infected_prevalence, height=0.02)  # Adjust threshold if needed

    # If at least one peak exists, capture the first peak's amplitude
    if len(peaks) > 0:
        first_peak_amplitude = infected_prevalence[peaks[0]]
    else:
        first_peak_amplitude = 0  # No peak detected at all

    # **If a second peak exists, prioritize it massively over single peaks**
    if len(peaks) >= 2:
        second_peak_amplitude = infected_prevalence[peaks[1]]

        # Find the trough between peaks
        trough_idx = np.argmin(infected_prevalence[peaks[0] : peaks[1]]) + peaks[0]
        trough_amplitude = infected_prevalence[trough_idx]

        # Score based on how close first and second peaks are (ideal ratio = 1)
        peak_ratio = abs(second_peak_amplitude / first_peak_amplitude - 1)

        # Measure "trough-iness" as how deep the valley is
        trough_depth = first_peak_amplitude - trough_amplitude

        # Penalize high first peaks (we want them low)
        first_peak_penalty = max(0, first_peak_amplitude - 0.05)  # Penalize if above 5% prevalence

        # **Significantly reduce score if a second peak is found**
        score = (peak_ratio - trough_depth + first_peak_penalty) * 0.1  # 10x better

        print(f"✅ Second peak found! {peak_ratio=}, {trough_depth=}, {first_peak_penalty=}, {score=}")
        return score

    # **Guiding Single-Peak Solutions Toward Two-Peak Solutions**
    # If no second peak exists, return a smoother penalty.
    # Lower first peaks are better, but we encourage later infection rebounds.
    post_peak_growth = np.max(np.diff(infected_prevalence[peaks[0] :])) if len(peaks) > 0 else 0

    # Score single-peak solutions: Lower peaks are rewarded, post-peak growth is encouraged
    penalty = (first_peak_amplitude * 50) - post_peak_growth * 100  # Encourage later rebounds

    print(f"❌ No second peak yet. {first_peak_amplitude=}, {post_peak_growth=}, {penalty=}")
    return max(5, penalty)  # Ensure values stay smooth and don't jump directly to "10"


def evaluate_3_weights(results_csv):
    """Evaluate if the I series of Node 0 has a second peak, while penalizing high first peaks."""

    # Load simulation results
    df = pd.read_csv(results_csv)

    # Filter for Node 0
    node_0_data = df[df["Node"] == 0]

    # Extract infected column and total population
    infected_series = node_0_data["Infected"].values
    total_population = node_0_data["Susceptible"] + node_0_data["Infected"] + node_0_data["Recovered"]

    # Normalize infected series to prevalence
    infected_prevalence = infected_series / total_population

    # Find peaks
    peaks, _ = find_peaks(infected_prevalence, height=0.04)  # Adjust threshold as needed

    # If at least one peak exists, capture the first peak's amplitude
    if len(peaks) > 0:
        infected_prevalence = infected_prevalence.to_numpy()  # Ensure NumPy indexing
        first_peak_amplitude = infected_prevalence[peaks[0]]
    else:
        first_peak_amplitude = 0  # No peak detected at all

    # If fewer than 2 peaks, return a continuous penalty rather than an early exit
    if len(peaks) < 2:
        return 10 + first_peak_amplitude * 100  # Penalize high first peaks more

    # Extract the second peak's amplitude
    second_peak_amplitude = infected_prevalence[peaks[1]]

    # Find the trough between peaks
    trough_idx = np.argmin(infected_prevalence[peaks[0] : peaks[1]]) + peaks[0]
    trough_amplitude = infected_prevalence[trough_idx]

    # Score based on how close first and second peaks are (ideal ratio = 1)
    peak_ratio = abs(second_peak_amplitude / first_peak_amplitude - 1)

    # Measure "trough-iness" as how deep the valley is
    trough_depth = first_peak_amplitude - trough_amplitude

    # Penalize high first peaks (we want them low)
    first_peak_penalty = max(0, first_peak_amplitude - 0.05)  # Penalize if above 5% prevalence

    # Final score (lower is better)
    score = peak_ratio - trough_depth + first_peak_penalty

    print(f"{peak_ratio=}, {trough_depth=}, {first_peak_penalty=}, {score=}")

    return score


def evaluate_second_peak_no_trough(results_csv):
    """Evaluate if the I series of Node 0 has a second peak based on amplitude ratio."""

    # Load simulation results
    df = pd.read_csv(results_csv)

    # Filter for Node 0
    node_0_data = df[df["Node"] == 0]

    # Compute total population at each timestep
    total_population = node_0_data["Susceptible"] + node_0_data["Infected"] + node_0_data["Recovered"]

    # Normalize infected column
    infected_series = node_0_data["Infected"] / total_population

    # Find peaks with a minimum height threshold of 0.04
    peaks, _ = find_peaks(infected_series, height=0.04)  # Adjust height threshold to filter out noise

    if len(peaks) >= 2:
        # Get the amplitudes of the first and second peaks
        first_peak_amplitude = infected_series.iloc[peaks[0]]
        second_peak_amplitude = infected_series.iloc[peaks[1]]
        # Calculate the ratio of the second peak to the first peak's amplitude
        amplitude_ratio = second_peak_amplitude / first_peak_amplitude

        # We want the amplitudes to be as close as possible, so minimize the absolute difference from 1
        return -abs(amplitude_ratio - 1)  # The closer the ratio is to 1, the better

    return float("inf")  # Return a high value if there are not enough peaks


# Paths to input/output files
PARAMS_FILE = "params.json"
RESULTS_FILE = "simulation_results.csv"


def objective(trial):
    """Optuna objective function that runs laser.py with trial parameters and evaluates results."""

    Path.unlink(RESULTS_FILE)

    # Suggest values for calibration
    migration_rate = trial.suggest_float("migration_rate", 0.0001, 0.01)
    transmission_rate = trial.suggest_float("transmission_rate", 0.05, 0.5)
    # migration_rate = trial.suggest_float("migration_rate", 0.004, 0.004)
    # transmission_rate = trial.suggest_float("transmission_rate", 0.145, 0.145)

    # Set up parameters
    params = {
        "population_size": 500_000,
        "nodes": 20,
        "timesteps": 500,
        "initial_infected_fraction": 0.01,
        "transmission_rate": transmission_rate,
        "migration_rate": migration_rate,
    }

    # Write parameters to JSON file
    with Path(PARAMS_FILE).open("w") as f:
        json.dump(params, f, indent=4)

    def get_docker_runstring():
        # cmd = f"docker run --rm -v .:/app/shared docker.io/jbloedow/my-laser-app:latest"
        cmd = "docker run --rm -v .:/app/shared my-laser-app:latest"
        return cmd.split()

    def get_native_runstring():
        return [sys.executable, str(laser_script)]

    print(f"Will be looking for {RESULTS_FILE}")
    # Run laser.py as a subprocess
    try:
        # Run the model 4 times and collect results
        scores = []
        for _ in range(4):
            subprocess.run(get_docker_runstring(), check=True)
            # subprocess.run(["python", "laser.py"], check=True)

            # Wait until RESULTS_FILE is written
            while not Path(RESULTS_FILE).exists():
                time.sleep(0.1)

            score = evaluate_3_weights_better(RESULTS_FILE)  # Evaluate results
            scores.append(score)

        # Return the average score
        return np.mean(scores)

    except subprocess.CalledProcessError as e:
        print(f"Error running laser.py: {e}")
        return float("inf")  # Penalize failed runs
