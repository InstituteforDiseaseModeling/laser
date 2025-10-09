# Optimize custom models

As an agent-based model, research using LASER will require thousands of simulation replicates. While the model is designed to perform well with large agent populations, there is still a need to utilize high compute power and to optimize model performance.

When creating custom models, knowing how to identify and fix performance bottlenecks can save compute time and speed results.

## Identifying bottlenecks

Typically, we do not recommend running the code through a profiler, at least not initially. Instead, we take advantage of LASER’s highly modular structure and AI-driven optimization.

The first step is to add simple timing code that tracks the total time spent in each component over a realistic simulation. Then, plot a pie chart at the end to visualize where the most time is spent. A simple way to track execution time is using the `time package`.

/// details | Code example: Identify bottlenecks

```
def run(self):
    self.component_times = {component.__class__.__name__: 0 for component in self.components}
    self.component_times["reporting"] = 0
    for tick in tqdm(range(self.pars.timesteps)):
        for component in self.components:
            start_time = time.time()  # Start timing for the component
            component.step()
            end_time = time.time()  # End timing for the component

            # Accumulate the time taken for this component
            elapsed_time = end_time - start_time
            component_name = component.__class__.__name__
```
///

This often reveals the top 1 to 3 performance bottlenecks. Focus first on the biggest offender—it provides the most opportunity for speedup. Often, the largest bottleneck is not what you might instinctively expect. Avoid optimizing a component only to find out it contributes just a small percentage of the total runtime. A modest improvement in the runtime of an “expensive” component is often more effective than spending a lot of time on highly optimizing a component which only accounts for a small fraction of runtime. Also, make sure that your reporting code is being measured and reported, ideally in its own ‘bucket’. This may be easier or harder depending on how you are doing reporting. Since reporting usually involves counting over the entire population, it usually shows up as a hotspot sooner or later. Fortunately, it’s usually fairly easy to speed up. Or even eliminate.


## Leverage AI

Once you have identified the slowest component, the easiest way to improve performance is by using ChatGPT. Try prompting with:

`"This code is much too slow. (My arrays are all about 1e6 or 1e7 in size.)"`

If your code consists mainly of for-loops without much NumPy, you can add:

`"Is there anything we can vectorize better with NumPy?"`

This approach can often transform a naive implementation into a highly optimized one.



## Implement unit tests

Instead of testing performance within the full simulation, consider building unit tests. This ensures correctness while optimizing for speed.

- Use AI to generate unit tests that validate output against a known correct (but slower) version.
- Include performance benchmarks in the tests.
- Ensure large array sizes (e.g., 1 million+ elements) to get meaningful speed comparisons.

<!-- would be nice if we have unit tests already built? Things that are more concrete to share? -->


## Optimize with NumPy and Numba

After achieving good performance with NumPy, consider trying Numba for further improvements.

Even if you’re new to Numba, ChatGPT can generate optimized solutions easily. Keep in mind:

- Numba moves back to explicit for-loops (unlike NumPy, which uses vectorization syntax).
- GPT’s first solution may use `range` instead of `prange`. Prompt it with:
    `"Can we parallelize this with prange?"`
- If your code involves common counters, atomic operations may become a bottleneck. Ask GPT about:
    `"Can we use thread-local storage to avoid atomic operations?"`
- Numba may be slower than NumPy for small arrays (e.g., thousands or tens of thousands of elements). Test with at least 1 million elements.


## C and OpenMP

If the best Numba solution still isn’t fast enough, consider compiled C.

- Use ctypes to call C functions from Python.
- Mention “use OpenMP” in AI prompts if parallelization is possible.
- Ask: `"Can you generate an OpenMP solution with the best pragmas?"`
- The more CPU cores available, the greater the potential speedup. That said, it’s usually a case of diminishing returns as one goes from 8 cores to 16 and to 32. Our research shows that often you’re better off running 4 sims across 8 cores each than running 1 sim on all 32 cores available. Also be aware that with both Numba and OpenMP you can constrain the number of cores used to less than the number available by setting the appropriate environment variable. (Numba environment variable = NUMBA_NUM_THREADS; OpenMP environment variable = OMP_NUM_THREADS)

## Additional advice

- Don’t duplicate. Sometimes reporting will duplicate transmission code and need to be combined.

- Never append. There may be cases where you are collecting information as it happens without knowing ahead of time how many rows/entries/elements you’ll need. This is easy in Python using list appending, for example, but that’s a performance killer. Really try to find a way to figure out ahead of time how many entries there will be, and then allocate memory for that, and insert into the existing row.

- Some components have long time-scales, like mortality. By default you are probably going to end up doing most component steps every timestep. You can probably get away with doing mortality updates, for example, far less often. You can experiement with weekly, fortnightly or monthly updates, depending on the timescale of the component you’re optimizing. Just be sure to move everything forward by a week if you’re only doing the update every week. And expect “blocky” plots. Note that there are fancier solutions like ‘strided sharding’ (details omitted).

- When prompting AI, use questions rather than directives. For example:

    `"Do you think it might be better to...?"`

    This prevents oversteering the AI into suboptimal solutions.