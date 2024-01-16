#! /usr/bin/env python3

"""Explore the final attack fraction of the SEIR model."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from test_agentseir import test_seir


class Namespace:
    """A simple class for storing parameters."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def batch(meta):
    """Run (optionally) a batch of simulations and, optionally, plot the results."""
    meta.resultsdir.mkdir(exist_ok=True)

    if meta.simulate:
        print(f"Running {meta.steps} simulations with {meta.replicates} replicates each. Results in '{meta.resultsdir}'")
        reproductive_number = np.linspace(meta.min_r0, meta.max_r0, meta.steps)
        for r0 in reproductive_number:
            for replicate in range(meta.replicates):
                params = Namespace(
                    timesteps=365,
                    pop_size=1_000_000,
                    exp_mean=4.0,
                    exp_std=1.0,
                    inf_mean=5.0,
                    inf_std=1.0,
                    initial_inf=10,
                    r_naught=r0,
                    poisson=True,
                    vaccinate=False,
                    masking=False,
                    seed=datetime.now().microsecond,  # noqa: DTZ005
                    filename=meta.resultsdir / f"{r0:0.2f}_{replicate}.csv",
                    beta=r0 / 5.0,
                )
                test_seir(params)

    if meta.plot:
        print(f"Reading results from '{meta.resultsdir}'")
        points = []
        for filename in meta.resultsdir.glob("*.csv"):
            df = pd.read_csv(filename)
            r_naught = float(filename.name.split("_")[0])
            final = df.iloc[-1]
            fraction = final.recovered / (final.susceptible + final.exposed + final.infected + final.recovered)
            points.append((r_naught, fraction))

        points = pd.DataFrame(points, columns=["R0", "fraction"])
        print(f"Writing scatter plot to '{meta.resultsdir / meta.plotname}'")
        points.plot.scatter(x="R0", y="fraction", figsize=(12, 9)).get_figure().savefig(meta.resultsdir / meta.plotname)

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nosimulate", dest="simulate", action="store_false")
    parser.add_argument("--noplot", dest="plot", action="store_false")
    parser.add_argument("--resultsdir", type=Path, default=Path(__file__).parent / "results_seir")
    parser.add_argument("--min_r0", type=float, default=0.5)
    parser.add_argument("--max_r0", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--plotname", default="seir_scatter.png")
    args = parser.parse_args()
    batch(args)
