"""Utility functions for the IDMLaser package."""

from collections.abc import Iterable
from json import JSONEncoder
from pathlib import Path
from typing import Any
from typing import Tuple
from typing import Union

import numpy as np


class PriorityQueue:
    """A priority queue implemented using a heap."""

    def __init__(self, capacity, dtype=np.uint32):
        self.payloads = np.zeros(capacity, dtype=dtype)
        self.priority = np.zeros(capacity, dtype=np.uint32)
        self.size = 0

    def push(self, payload, priority):
        _pq_push(self, payload, priority)

    def peek(self) -> Tuple[Any, np.uint32]:
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return (self.payloads[0], self.priority[0])

    def pop(self) -> Tuple[Any, np.uint32]:
        return _pq_pop(self)

    def __len__(self):
        return self.size


def _pq_push(pq: PriorityQueue, payload: Any, priority: np.uint32):
    """Push an item with a priority into the priority queue."""
    if pq.size >= len(pq.payloads):
        raise IndexError("Priority queue is full")
    pq.payloads[pq.size] = payload
    pq.priority[pq.size] = priority
    pq.size += 1
    index = pq.size - 1
    while index > 0:
        parent_index = (index - 1) // 2
        if pq.priority[index] < pq.priority[parent_index]:
            pq.payloads[index], pq.payloads[parent_index] = pq.payloads[parent_index], pq.payloads[index]
            pq.priority[index], pq.priority[parent_index] = pq.priority[parent_index], pq.priority[index]
            index = parent_index
        else:
            break

    return


def _pq_pop(pq: PriorityQueue) -> Tuple[Any, np.uint32]:
    """Remove the item with the highest priority from the priority queue."""
    if pq.size == 0:
        raise IndexError("Priority queue is empty")

    payload, priority = pq.peek()

    pq.payloads[0] = pq.payloads[pq.size - 1]
    pq.priority[0] = pq.priority[pq.size - 1]
    pq.size -= 1

    index = 0
    while index < pq.size:
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        smallest = index

        if left_child_index < pq.size and pq.priority[left_child_index] < pq.priority[smallest]:
            smallest = left_child_index

        if right_child_index < pq.size and pq.priority[right_child_index] < pq.priority[smallest]:
            smallest = right_child_index

        if smallest != index:
            pq.payloads[index], pq.payloads[smallest] = pq.payloads[smallest], pq.payloads[index]
            pq.priority[index], pq.priority[smallest] = pq.priority[smallest], pq.priority[index]
            index = smallest
        else:
            break

    return payload, priority


def pop_from_cbr_and_mortality(
    initial: Union[int, np.integer],
    cbr: Union[Iterable[Union[int, float, np.number]], Union[int, float, np.number]],
    mortality: Union[Iterable[Union[int, float, np.number]], Union[int, float, np.number]],
    nyears: Union[int, np.integer],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate births, deaths, and total population for each year."""
    births = np.zeros(nyears, dtype=np.uint32)
    deaths = np.zeros(nyears, dtype=np.uint32)
    population = np.zeros(nyears + 1, dtype=np.uint32)

    if not isinstance(cbr, Iterable):
        cbr = np.full(nyears, cbr)
    if not isinstance(mortality, Iterable):
        mortality = np.full(nyears, mortality)

    population[0] = initial

    for year, cbr_i, mortality_i in zip(range(nyears), cbr, mortality):
        current_population = population[year]
        births[year] = np.random.poisson(cbr_i * current_population / 1000)
        deaths[year] = np.random.poisson(mortality_i * current_population / 1000)
        population[year + 1] = population[year] + births[year] - deaths[year]

    return births, deaths, population


def daily_births_deaths_from_annual(annual_births, annual_deaths) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate daily births and deaths from annual values."""
    assert len(annual_births) == len(annual_deaths), "Lengths of arrays must be equal"
    nyears = len(annual_births)
    daily_births = np.zeros(nyears * 365, dtype=np.uint32)
    daily_deaths = np.zeros(nyears * 365, dtype=np.uint32)

    for day in range(nyears * 365):
        year = day // 365
        doy = (day % 365) + 1
        daily_births[day] = (annual_births[year] * doy // 365) - (annual_births[year] * (doy - 1) // 365)
        daily_deaths[day] = (annual_deaths[year] * doy // 365) - (annual_deaths[year] * (doy - 1) // 365)

    return daily_births, daily_deaths


class NumpyJSONEncoder(JSONEncoder):
    """Custom JSON encoder for NumPy arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return JSONEncoder.default(self, obj)


# Derived from table 1 of "National Vital Statistics Reports Volume 54, Number 14 United States Life Tables, 2003" (see README.md)
cumulative_deaths = [
    0,
    687,
    733,
    767,
    792,
    811,
    829,
    844,
    859,
    872,
    884,
    895,
    906,
    922,
    945,
    978,
    1024,
    1081,
    1149,
    1225,
    1307,
    1395,
    1489,
    1587,
    1685,
    1781,
    1876,
    1968,
    2060,
    2153,
    2248,
    2346,
    2449,
    2556,
    2669,
    2790,
    2920,
    3060,
    3212,
    3377,
    3558,
    3755,
    3967,
    4197,
    4445,
    4715,
    5007,
    5322,
    5662,
    6026,
    6416,
    6833,
    7281,
    7759,
    8272,
    8819,
    9405,
    10032,
    10707,
    11435,
    12226,
    13089,
    14030,
    15051,
    16146,
    17312,
    18552,
    19877,
    21295,
    22816,
    24445,
    26179,
    28018,
    29972,
    32058,
    34283,
    36638,
    39108,
    41696,
    44411,
    47257,
    50228,
    53306,
    56474,
    59717,
    63019,
    66336,
    69636,
    72887,
    76054,
    79102,
    81998,
    84711,
    87212,
    89481,
    91501,
    93266,
    94775,
    96036,
    97065,
    97882,
    100000,
]

__cdnp = np.array(cumulative_deaths)


def predicted_year_of_death(age_years, max_year=100):
    """
    Calculates the predicted year of death based on the given age in years.

    Parameters:
    - age_years (int): The age of the individual in years.
    - max_year (int): The maximum year to consider for calculating the predicted year of death. Default is 100.

    Returns:
    - yod (int): The predicted year of death.

    Example:
    >>> predicted_year_of_death(40, max_year=80)
    62
    """

    # e.g., max_year == 10, 884 deaths are recorded in the first 10 years
    total_deaths = __cdnp[max_year + 1]
    # account for current age, i.e., agent is already 4 years old, so 792 deaths have already occurred
    already_deceased = __cdnp[age_years]
    # this agent will be one of the deaths in (already_deceased, total_deaths] == [already_deceased+1, total_deaths+1)
    draw = np.random.randint(already_deceased + 1, total_deaths + 1)
    # find the year of death, e.g., draw == 733, searchsorted("left") will return 2, so the year of death is 1
    yod = np.searchsorted(__cdnp, draw, side="left") - 1

    return yod


def predicted_day_of_death(age_days, max_year=100):
    """
    Calculates the predicted day of death based on the given age in days and the maximum year of death.

    Parameters:
    - age_days (int): The age in days.
    - max_year (int): The maximum year of death. Defaults to 100.

    Returns:
    - dod (int): The predicted day of death.

    The function first calculates the predicted year of death based on the given age in days and the maximum year of death.
    Then, it randomly selects a day within the year of death.
    The age/date of death has to be greater than today's age.
    Finally, it calculates and returns the predicted day of death.

    Note: This function assumes that there are 365 days in a year.
    """

    yod = predicted_year_of_death(age_days // 365, max_year)

    # if the death age year is not the current age year pick any day that year
    if age_days // 365 < yod:
        # the agent will die sometime in the year of death, so we randomly select a day
        doy = np.random.randint(365)
    else:
        # the agent will die on or before next birthday
        age_doy = age_days % 365  # 0 ... 364
        if age_doy < 364:
            # there is time before the next birthday, pick a day at random
            doy = np.random.randint(age_doy + 1, 365)
        else:
            # the agent's birthday is tomorrow; bummer of a birthday present
            yod += 1
            doy = 0

    dod = yod * 365 + doy

    return dod
