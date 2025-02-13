===============================
LASER Performance Optimization
===============================

It's fairly easy to generate working LASER model code that uses NumPy and performs well for populations of 100,000 agents. We expect LASER code to perform efficiently for **200,000,000 agents** (or more). The challenge is how to get there.

Identifying Performance Bottlenecks
-----------------------------------

Typically, we do **not** recommend running the code through a profiler—at least not initially. Instead, we take advantage of **LASER's highly modular structure** and **AI-driven optimization**.

The first step is to **add simple timing code** that tracks the total time spent in each component over a realistic simulation. Then, **plot a pie chart** at the end to visualize where the most time is spent. A simple way to track execution time is using the ``time`` package. Example code:

.. code-block:: text

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

This often reveals **the top 1 to 3 performance bottlenecks**. Focus first on the biggest offender—it provides the most opportunity for speedup. Often, the largest bottleneck is **not** what you might instinctively expect. Avoid optimizing a component only to find out it contributes just **a small percentage of the total runtime**. A modest improvement in the runtime of an "expensive" component is often more effective than spending a lot of time on highly optimizing a component which only accounts for a small fraction of runtime. Also, make sure that your reporting code is being measured and reported, ideally in its own 'bucket'. This may be easier or harder depending on how you are doing reporting. Since reporting usually involves counting over the entire population, it usually shows up as a hotspot sooner or later. Fortunately, it's usually fairly easy to speed up. Or even eliminate.

Leveraging AI for Code Optimization
-----------------------------------

Once you've identified the **slowest component**, the easiest way to improve performance is by using **ChatGPT**. Try prompting with:

.. code-block:: text

   "This code is much too slow. (My arrays are all about 1e6 or 1e7 in size.)"

If your code consists mainly of **for-loops** without much **NumPy**, you can add:

.. code-block:: text

   "Is there anything we can vectorize better with NumPy?"

This approach can often **transform a naive implementation into a highly optimized one**.

Unit Tests for Performance and Accuracy
---------------------------------------

Instead of testing performance within the full simulation, consider building **unit tests**. This ensures **correctness** while optimizing for **speed**.

- Use AI to generate **unit tests** that validate output against a known correct (but slower) version.
- Include **performance benchmarks** in the tests.
- Ensure **large array sizes** (e.g., **1 million+ elements**) to get meaningful speed comparisons.

Optimizing with NumPy and Numba
-------------------------------

After achieving good performance with **NumPy**, consider trying **Numba** for further improvements.

Even if you're new to Numba, **ChatGPT** can generate optimized solutions easily. Keep in mind:

- **Numba moves back to explicit for-loops** (unlike NumPy, which uses vectorization syntax).

- GPT's first solution may use ``range`` instead of ``prange``. Prompt it with:

  .. code-block:: text

     "Can we parallelize this with prange?"

- If your code involves **common counters**, **atomic operations** may become a bottleneck.
  Ask GPT about:

  .. code-block:: text

     "Can we use thread-local storage to avoid atomic operations?"

- **Numba may be slower than NumPy for small arrays** (e.g., thousands or tens of thousands of elements). Test with **at least 1 million elements**.

Further Tricks
^^^^^^^^^^^^^^

- Don't duplicate: Sometimes reporting will duplicate transmission code and need to be combined.
- Never append. There may be cases where you are collecting information as it happens without knowing ahead of time how many rows/entries/elements you'll need. This is easy in Python using list appending, for example, but that's a performance killer. Really try to find a way to figure out ahead of time how many entries there will be, and then allocate memory for that, and insert into the existing row.
- Some components have long time-scales, like mortality. By default you are probably going to end up doing most component steps every timestep. You can probably get away with doing mortality updates, for example, far less often. You can experiement with weekly, fortnightly or monthly updates, depending on the timescale of the component you're optimizing. Just be sure to move everything forward by a week if you're only doing the update every week. And expect "blocky" plots. Note that there are fancier solutions like 'strided sharding' (details omitted).

When **prompting AI**, use **questions rather than directives**. Example:

.. code-block:: text

   "Do you think it might be better to...?"

This prevents oversteering the AI into suboptimal solutions.

Beyond Numba: C and OpenMP
--------------------------

If the best **Numba** solution still isn't fast enough, consider **compiled C**.

- Use **ctypes** to call C functions from Python.
- Mention **"use OpenMP"** in AI prompts if parallelization is possible.
- Ask:

  .. code-block:: text

     "Can you generate an OpenMP solution with the best pragmas?"

- The more CPU cores available, the **greater the potential speedup**. That said, it's usually a case of diminishing returns as one goes from 8 cores to 16 and to 32. Our research shows that often you're better off running 4 sims across 8 cores each than running 1 sim on all 32 cores available. Also be aware that with both Numba and OpenMP you can constrain the number of cores used to less than the number available by setting the appropriate environment variable. (Numba environment variable = NUMBA_NUM_THREADS; OpenMP environment variable = OMP_NUM_THREADS)

Advanced Hardware-Dependent Performance Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond compiled C extensions using OpenMP for parallelization across CPU cores, even greater performance gains can be achieved by leveraging hardware-specific optimizations.

SIMD
----

Modern CPUs include low-level **Single Instruction, Multiple Data (SIMD)** instruction sets that enable direct access to vectorized operations. While compilers attempt to generate optimal SIMD instructions automatically, they are not always perfect.

In theory, writing SIMD code manually can lead to significant performance gains, but this requires deep, architecture-specific knowledge. Fortunately, AI-assisted development tools can help generate such code. However, in practice, achieving meaningful speedups for complex use cases remains challenging. Additionally, since SIMD instruction sets vary by hardware, code optimized for a development machine may not work on a different target machine. Consult a developer to determine applicability for your use case.

GPU
---

GPUs can provide massive speedups when used effectively, but several challenges must be considered:

- GPU hardware must be available on the target machine.
- GPU-specific code needs to be written, often using CUDA (for NVIDIA GPUs) or other frameworks like OpenCL or ROCm.
- The overhead of transferring data between CPU and GPU memory can negate performance benefits unless the system has unified memory.

We continue to explore GPU acceleration for LASER, particularly for cases where computational workloads justify the overhead of GPU execution.


Final Thoughts
--------------

In some cases, an algorithm may be **inherently sequential**, meaning **parallelization won’t help**. Be mindful that AI might not always indicate when you're **hitting a fundamental limitation**.

By following this process—**profiling via timing, leveraging AI, and incrementally optimizing with NumPy, Numba, and C**—you can take LASER models from **functional** to **high-performance** at massive scales.
