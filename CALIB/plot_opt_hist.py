import optuna
from optuna.visualization import plot_optimization_history

study = optuna.load_study(study_name="spatial_demo_calibration2", storage="sqlite:///optuna_study.db")
fig = plot_optimization_history(study)
fig.show()
