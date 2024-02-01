## Profiling

### Run Profiler

```python3 -m cProfile -o measles.prof ./consolidate.py -p 1000000 --exp_mean 8 --exp_std 2 --inf_mean 8 --inf_std 1 --r_naught 10 --filename measles.csv```

### Get Profile Information

```python3 -c "import pstats; pstats.Stats('measles.prof').sort_stats('cumulative').print_stats()" > measles.txt```

## Plotting

```python
filename = "measles2050"
import pandas as pd
df = pd.read_csv(f"{filename}.csv")
# import matplotlib.pyplot as plt
# f = plt.figure(figsize=(16,12), dpi=300)
# ax = f.subplots()
ax = df.plot("timestep", ["susceptible", "recovered"], title="GroupedSEIR", figsize=(16,12))
ax2 = ax.twinx()
df.plot("timestep", ["exposed", "infectious"], ax=ax2)
# f.savefig(f"{filename}.png")
ax.get_figure().tight_layout()
ax.get_figure().savefig(f"{filename}.png")
```
