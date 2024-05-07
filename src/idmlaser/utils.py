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
