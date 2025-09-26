# Loading Data and Initializing Populations

<!-- Need to add an introduction here.-->


## Loading data

## Initializing populations

## Squashing, saving, and loading

As the number agents in your LASER population model grows (e.g., 1e8), it can become computationally expensive and unnecessary to repeatedly run the same initialization routine every sim. In many cases -— particularly during model calibration -— it is far more efficient to initialize the population once, save it, and then reload the initialized state for subsequent runs.

This approach is especially useful when working with EULAs – Epidemiologically Uninteresting Light Agents. For example, it can be a very powerful optimization to compress all the agents who are already (permanently) recovered or immune in a measles or polio model into a number/bucket. In such models, the majority of the initial population may be in the “Recovered” state, potentially comprising 90% or more of all agents. If you are simulating 100 million agents, storing all of them can result in punitive memory usage.

To address this, LASER supports a **squashing** process. Squashing involves defragmenting the data frame such that all epidemiologically active or “interesting” agents (e.g., Susceptible or Infectious) are moved to the beginning of the array or table, and less relevant agents (e.g., Recovered) are moved to the end. Though please note that you should assume that squashed agent data is overwritten.

Some notes about squashing:

- The population count is adjusted so that all for loops and step functions iterate only over the active population.
- This not only reduces memory usage but also improves performance by avoiding unnecessary computation over inactive agents.


Some notes about using saved populations:

- You will want to be confident that the saved population is sufficiently randomized and representative of your overall population.
- If you are calibrating parameters used to create the initial population in the first place, you’ll need to recreate those parts of the population after loading, diminishing the benefit of the save/load approach.
- When saving a **snapshot**, note that only the active (unsquashed) portion of the population is saved. Upon reloading:

    - Only this subset is allocated in memory.
    - This prevents the performance penalty of managing large volumes of unused agent data.

!!! note
    Before squashing, you should count and record the number of recovered (or otherwise squashed) agents. This count should be stored in a summary variable —- typically the R column of the results data frame. This ensures your model retains a complete epidemiological record even though the agents themselves are no longer instantiated.

Procedure:

1. Add squashing:
    - Add a `squash_recovered()` function. This should call `LaserFrame.squash(…)` with a boolean mask that includes non-recovered agents (disease_state != 2). You may choose a different criterion, such as age-based squashing.
    - Count your “squashed away” agents first. You must compute and store all statistics related to agents being squashed before the `squash()` call. After squashing, only the left-hand portion of the arrays (up to .count) remains valid.
    - Seed infections after squashing. If your model seeds new infections (disease_state == 1), this must happen after squashing. Otherwise, infected agents may be inadvertently removed.
    - Store the squashed-away totals by node. Before squashing, compute and record node-wise totals (e.g., recovered counts) in `results.R[0, :]` so this pre-squash information persists.
    - (Optionally) simulate EULA effects once and save. If modeling aging or death among squashed agents, simulate this up front and store the full `[time, node]`` matrix (e.g., `results.R[:, :]`). This avoids recomputation at runtime.

2. Save function: implement a `save(path)` method:
    - Use `LaserFrame.save_snapshot(path, results_r=..., pars=...)`
    - Include:
        - The squashed population (active agents only)
        - The `results.R` matrix containing both pre-squash and live simulation values
        - The full parameter set in a `PropertySet`

3. Load function: implement a `load(path)` class method:
    - Call `LaserFrame.load_snapshot(path)` to retrieve:
        - Population frame
        - Results matrix
        - Parameters
    - Set `.capacity = .count` if not doing births, else set capacity based on projected population growth from count.
    - Reconstruct all components using `init_from_file()`

    !!! warning
        When modeling vital dynamics, especially births, there is an additional step needed to ensure consistency after loading:

        Property initialization for unborn individuals must be repeated if your model pre-assigns properties up to `.capacity`. For example, if timers or demographic attributes (like `date_of_birth`) are pre-initialized at `t=0`, you must ensure this initialization is re-applied after loading, because only the `.count` population is reloaded, not the future `.capacity`.

        Failing to do so may result in improperly initialized agents being birthed after the snapshot load, which can lead to subtle or catastrophic model errors.

4. Preserve EULA'd results:

    Use "+=" to track new recoveries alongside pre-squash R values. In `run()`, use additive updates so that pre-saved recovered agents are preserved:

    ```
    self.results.R[t, nid] += ((self.population.node_id == nid) &
                           (self.population.disease_state == 2)).sum()
    ```

    This ensures your output accounts for both squashed-away immunity and recoveries during the live simulation.

A complete example of adding squashing, saving, and loading to SIR models is available in the [tutorials](../tutorials/squashing-sir.md).

<!--
## Other pre-run tasks
-->
