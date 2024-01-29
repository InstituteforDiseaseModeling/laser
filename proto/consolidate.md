## Profiling

### Run Profiler

```python3 -m cProfile -o measles.prof ./consolidate.py -p 1000000 --exp_mean 8 --exp_std 2 --inf_mean 8 --inf_std 1 --r_naught 10 --filename measles.csv```

### Get Profile Information

```python3 -c "import pstats; pstats.Stats('measles.prof').sort_stats('cumulative').print_stats()" > measles.txt```
