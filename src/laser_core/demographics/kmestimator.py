"""
This module provides the KaplanMeierEstimator class for predicting the year and age at death based on given ages and cumulative death data.

Classes:

    - KaplanMeierEstimator: A class to perform Kaplan-Meier estimation for predicting the year and age at death.

Functions:

    - _pyod(ages_years: np.ndarray, cumulative_deaths: np.ndarray, max_year: np.uint32 = 100): Calculate the predicted year of death based on the given ages in years.

    - _pdod(age_in_days: np.ndarray, year_of_death: np.ndarray, day_of_death: np.ndarray): Calculate the predicted day of death based on the given ages in days and predicted years of death.

Usage example:

.. code-block:: python

    estimator = KaplanMeierEstimator(cumulative_deaths=np.array([...]))
    year_of_death = estimator.predict_year_of_death(np.array([40, 50, 60]), max_year=80)
    age_at_death = estimator.predict_age_at_death(np.array([40*365, 50*365, 60*365]), max_year=80)
"""

from pathlib import Path
from typing import Any
from typing import Union

import numba as nb
import numpy as np


class KaplanMeierEstimator:
    def __init__(self, source: Union[np.ndarray, list, Path, str]) -> None:
        """
        Initializes the KMEstimator with the given source data.

        Parameters:

            source : Union[np.ndarray, list, Path, str]

                The source data for the KMEstimator. It can be:

                - A numpy array of unsigned 32-bit integers.

                - A list of integers.

                - A Path object pointing to a file containing the data.

                - A string representing the file path.

        Raises:

            FileNotFoundError

                If the provided file path does not exist or is not a file.

            TypeError

                If the source type is not one of the accepted types (np.ndarray, list, Path, str).

            Value Error

                If the source inputs contain negative values or are not monotonically non-decreasing.

        Notes:

            - If the source is a file path, the file should contain comma-separated values with the data in the second column.

            - The source data is converted to a numpy array of unsigned 32-bit integers.
        """

        tsource = type(source)
        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            if not source.exists() or not source.is_file():
                raise FileNotFoundError(f"File not found: {source}")
            with source.open("r") as file:
                source = np.loadtxt(file, delimiter=",", usecols=1).astype(np.uint32)
        if isinstance(source, list):
            source = np.array(source, dtype=np.uint32)
        if not isinstance(source, np.ndarray):
            raise TypeError(f"Invalid source type: {tsource}")

        if not np.all(source >= 0):
            raise ValueError(f"Input values should be >= 0:\n\t{source}")

        if not np.all(np.diff(source.astype(np.int32)) >= 0):
            raise ValueError(f"Input values should be monotonically non-decreasing:\n\t{source}")

        self._cumulative_deaths = np.insert(source.astype(np.uint32), 0, 0)

        return

    @property
    def cumulative_deaths(self) -> np.ndarray:
        """
        Returns the original source data.
        """
        return self._cumulative_deaths[1:]  # exclude the leading zero

    def predict_year_of_death(self, ages_years: np.ndarray[Any, np.dtype[np.integer]], max_year: np.uint32 = 100) -> np.ndarray:
        """
        Calculate the predicted year of death based on the given ages in years.

        Parameters:

            ages_years (np.ndarray): The ages of the individuals in years.

            max_year (int): The maximum year to consider for calculating the predicted year of death. Default is 100.

        Returns:

            year_of_death (np.ndarray): The predicted years of death.

        Example:

        .. code-block:: python

            predict_year_of_death(np.array([40, 50, 60]), max_year=80) # returns something like array([62, 72, 82])
        """

        assert np.all(ages_years <= max_year), f"{ages_years.max()=} is not less than {max_year=}"
        year_of_death = _pyod(ages_years, self._cumulative_deaths, np.uint32(max_year))
        assert np.all(year_of_death <= max_year), f"{year_of_death.max()=} is not less than {max_year=}"

        return year_of_death

    def predict_age_at_death(self, ages_days: np.ndarray[Any, np.dtype[np.integer]], max_year: np.uint32 = 100) -> np.ndarray:
        """
        Calculate the predicted age at death (in days) based on the given ages in days.

        Parameters:

            ages_days (np.ndarray): The ages of the individuals in days.

            max_year (int): The maximum year to consider for calculating the predicted year of death. Default is 100.

        Returns:

            age_at_death (np.ndarray): The predicted days of death.

        Example:

        .. code-block:: python

            predict_age_at_death(np.array([40*365, 50*365, 60*365]), max_year=80) # returns something like array([22732, 26297, 29862])
        """

        assert np.all(ages_days < ((max_year + 1) * 365)), f"{ages_days.max()=} is not less than {((max_year + 1) * 365)=}"
        n = ages_days.shape[0]
        age_at_death = np.empty(n, dtype=ages_days.dtype)
        ages_years = np.empty(ages_days.shape, dtype=np.uint8)
        np.floor_divide(ages_days, 365, out=ages_years, casting="unsafe")
        year_of_death = _pyod(ages_years, self._cumulative_deaths, np.uint32(max_year))
        assert np.all(year_of_death <= max_year), f"{year_of_death.max()=} is not less than {max_year=}"

        _pdod(ages_days, year_of_death, age_at_death)

        # doy is now in age_at_death, add in the year
        age_at_death += year_of_death * 365
        # incoming individuals of age max. year + 364 days will die on the first day of the next year
        assert np.all(age_at_death < ((max_year + 1) * 365)), f"{age_at_death.max()=} is not <= {((max_year + 1) * 365)=}"

        return age_at_death


# Technically, we shouldn't allow any signed inputs because what would a negative age mean?
# int8 is the most limited, but 127 is enough for a reasonable maximum age in years.
# Separately, consider two versions of this function, one returning uint8 (for use in
# predict_year_of_death) and the other uint16 (for use in predict_age_at_death).
@nb.njit(
    [
        (nb.int8[:], nb.uint32[:], nb.uint32),
        (nb.int16[:], nb.uint32[:], nb.uint32),
        (nb.int32[:], nb.uint32[:], nb.uint32),
        (nb.int64[:], nb.uint32[:], nb.uint32),
        (nb.uint8[:], nb.uint32[:], nb.uint32),
        (nb.uint16[:], nb.uint32[:], nb.uint32),
        (nb.uint32[:], nb.uint32[:], nb.uint32),
        (nb.uint64[:], nb.uint32[:], nb.uint32),
        (nb.float32[:], nb.uint32[:], nb.uint32),
        (nb.float64[:], nb.uint32[:], nb.uint32),
    ],
    parallel=True,
)
def _pyod(ages_years: np.ndarray, cumulative_deaths: np.ndarray, max_year: np.uint32 = 100):  # pragma: no cover
    """
    Calculate the predicted year of death based on the given ages in years.

    Parameters:

        ages_years (np.ndarray): The ages of the individuals in years.

        cumulative_deaths (np.ndarray): Cumulative deaths by year.

        max_year (int): The maximum year to consider for calculating the predicted year of death. Default is 100.

    Returns:

        ysod (np.ndarray): The predicted years of death.

    Example:

    .. code-block:: python

        _pyod(np.array([40, 50, 60]), max_year=80) # returns something like array([62, 72, 82])
    """

    n = ages_years.shape[0]
    ysod = np.empty(n, dtype=np.uint16)  # _pdod needs this to be uint16
    total_deaths = cumulative_deaths[max_year + 1]

    for i in nb.prange(n):
        age_years = ages_years[i]
        already_deceased = cumulative_deaths[np.int8(age_years)]
        draw = np.random.randint(already_deceased + 1, total_deaths + 1)
        yod = np.searchsorted(cumulative_deaths, draw, side="left") - 1
        ysod[i] = yod

    return ysod


@nb.njit(
    [
        (nb.int16[:], nb.uint16[:], nb.int16[:]),
        (nb.int32[:], nb.uint16[:], nb.int32[:]),
        (nb.int64[:], nb.uint16[:], nb.int64[:]),
        (nb.uint16[:], nb.uint16[:], nb.uint16[:]),
        (nb.uint32[:], nb.uint16[:], nb.uint32[:]),
        (nb.uint64[:], nb.uint16[:], nb.uint64[:]),
        (nb.float32[:], nb.uint16[:], nb.float32[:]),
        (nb.float64[:], nb.uint16[:], nb.float64[:]),
    ],
    parallel=True,
)
def _pdod(age_in_days: np.ndarray, year_of_death: np.ndarray, day_of_death: np.ndarray):  # pragma: no cover
    n = age_in_days.shape[0]
    for i in nb.prange(n):
        age_days = np.uint32(age_in_days[i])
        if age_days // 365 < year_of_death[i]:
            # pick any day in the year of death
            day_of_death[i] = np.random.randint(365)
        else:
            age_doy = age_days % 365  # [0, 364] - day of year
            day_of_death[i] = np.random.randint(age_doy, 365)  # [age_doy, 364]

    return
