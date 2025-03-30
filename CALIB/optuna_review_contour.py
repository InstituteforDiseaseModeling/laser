import matplotlib.pyplot as plt
import optuna

# Load your study
study = optuna.load_study(study_name="spatial_demo_calibration4", storage="sqlite:///optuna_study.db")


def optuna_plots(study):
    # Plot parameter space
    optuna.visualization.plot_parallel_coordinate(study).show()
    optuna.visualization.plot_contour(study, params=["transmission_rate", "migration_rate"]).show()


def custom_plot(study):
    # Get trials and filter completed ones
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Extract parameter values and scores
    transmission_rates = [t.params["transmission_rate"] for t in trials]
    migration_rates = [t.params["migration_rate"] for t in trials]
    scores = [t.value for t in trials]  # Optuna minimizes by default; invert if needed

    # Scatter plot: color by score
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(transmission_rates, migration_rates, c=scores, cmap="viridis", edgecolors="k", alpha=0.8)
    plt.colorbar(sc, label="Objective Score")
    plt.xlabel("Transmission Rate")
    plt.ylabel("Migration Rate")
    plt.title("Optuna Parameter Search")
    plt.show()


def animate(trials):
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, t in enumerate(trials):
        plt.scatter(t.params["transmission_rate"], t.params["migration_rate"], color=plt.cm.viridis(i / len(trials)), alpha=0.7)

    plt.xlabel("Transmission Rate")
    plt.ylabel("Migration Rate")
    plt.title("Optuna Search Over Time")
    plt.show()


optuna_plots(study)
# custom_plot( study )
# animate( study.trials )
