"""Draw for date of death based on current age and the Kaplan-Meier curve."""

import numba as nb
import numpy as np

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

__cdnp = np.array(cumulative_deaths, dtype=np.uint32)


@nb.njit((nb.int32[:], nb.uint32, nb.uint32, nb.uint32[:]), parallel=True)
def _pysod(ages_years: np.ndarray, max_year: np.uint32 = 100, prng_seed: np.uint32 = 20240801, cdnp: np.ndarray = __cdnp):
    """
    Calculate the predicted year of death based on the given ages in years.

    Parameters:
    - ages_years (np.ndarray): The ages of the individuals in years.
    - max_year (int): The maximum year to consider for calculating the predicted year of death. Default is 100.

    Returns:
    - ysod (np.ndarray): The predicted years of death.

    Example:
    >>> pyod(np.array([40, 50, 60]), max_year=80)
    array([62, 72, 82])
    """

    n = ages_years.shape[0]
    ysod = np.empty(n, dtype=np.int32)

    for t in nb.prange(nb.get_num_threads()):
        np.random.seed(prng_seed + t)

    for i in nb.prange(n):
        age_years = ages_years[i]
        total_deaths = cdnp[max_year + 1]
        already_deceased = cdnp[age_years]
        draw = np.random.randint(already_deceased + 1, total_deaths + 1)
        yod = np.searchsorted(cdnp, draw, side="left") - 1
        ysod[i] = yod

    return ysod


def pysod(ages_years: np.ndarray, max_year: np.uint32 = 100, prng_seed: np.uint32 = 20240801, cdnp: np.ndarray = __cdnp):
    """
    Calculate the predicted year of death based on the given ages in years.

    Parameters:
    - ages_years (np.ndarray): The ages of the individuals in years.
    - max_year (int): The maximum year to consider for calculating the predicted year of death. Default is 100.
    - prng_seed (int): The seed for the random number generator. Default is 20240801.
    - cdnp (np.ndarray): Cumulative deaths by year.

    Returns:
    - ysod (np.ndarray): The predicted years of death.

    Example:
    >>> pyod(np.array([40, 50, 60]), max_year=80)
    array([62, 72, 82])
    """

    assert np.all(ages_years <= max_year), f"{ages_years.max()=} is not less than {max_year=}"
    ysod = _pysod(ages_years, max_year, prng_seed, cdnp)
    assert np.all(ysod <= max_year), f"{ysod.max()=} is not less than {max_year=}"

    return ysod


def predicted_year_of_death(age_years, max_year: int = 100, cdnp: np.ndarray = __cdnp):
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
    total_deaths = cdnp[max_year + 1]
    # account for current age, i.e., agent is already 4 years old, so 792 deaths have already occurred
    already_deceased = cdnp[age_years]
    # this agent will be one of the deaths in (already_deceased, total_deaths] == [already_deceased+1, total_deaths+1)
    draw = np.random.randint(already_deceased + 1, total_deaths + 1)
    # find the year of death, e.g., draw == 733, searchsorted("left") will return 2, so the year of death is 1
    yod = np.searchsorted(cdnp, draw, side="left") - 1
    assert 0 <= yod <= max_year, f"yod={yod} is not in [0, {max_year}]"

    return yod


@nb.njit((nb.int32[:], nb.int32[:], nb.int32[:], nb.uint32), parallel=True)
def _pdsod(ages_days: np.ndarray, ysod: np.ndarray, dods: np.ndarray, prng_seed: np.uint32 = 20240801):
    for t in nb.prange(nb.get_num_threads()):
        np.random.seed(prng_seed + t)

    n = ages_days.shape[0]
    for i in nb.prange(n):
        age_days = ages_days[i]
        if age_days // 365 < ysod[i]:
            # pick any day in the year of death
            dods[i] = np.random.randint(365)
        else:
            age_doy = age_days % 365  # [0, 364]
            if age_doy < 364:
                # pick any day between current day and end of year
                dods[i] = np.random.randint(age_doy + 1, 365)  # [age_doy+1, 364]
            else:
                # day of death is tomorrow, January 1st of next year
                ysod[i] += 1
                dods[i] = 0

    return


def pdsod(ages_days: np.ndarray, max_year: np.uint32 = 100, prng: np.random.Generator = np.random.default_rng()):
    """
    Calculate the predicted day of death based on the given ages in days.

    Parameters:
    - ages_days (np.ndarray): The ages of the individuals in days.
    - max_year (int): The maximum year to consider for calculating the predicted year of death. Default is 100.

    Returns:
    - dods (np.ndarray): The predicted days of death.

    Example:
    >>> pdod(np.array([40*365, 50*365, 60*365]), max_year=80)
    array([22732, 26297, 29862])
    """

    assert np.all(ages_days < ((max_year + 1) * 365)), f"{ages_days.max()=} is not less than {((max_year + 1) * 365)=}"
    n = ages_days.shape[0]
    dods = np.empty(n, dtype=np.int32)
    ysod = pysod(np.floor_divide(ages_days, 365, dtype=np.int32), np.uint32(max_year), np.uint32(prng.integers(0, 2**32)))
    assert np.all(ysod <= max_year), f"{ysod.max()=} is not less than {max_year=}"

    _pdsod(ages_days, ysod, dods, np.uint32(prng.integers(0, 2**32)))

    # doy is now in dods, add in the year
    dods += ysod * 365
    # incoming individuals of age max. year + 364 days will die on the first day of the next year
    assert np.all(dods <= ((max_year + 1) * 365)), f"{dods.max()=} is not <= {((max_year + 1) * 365)=}"

    return dods


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
