"""Class to store demographic data for agent-based models"""

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from datetime import timezone
from numbers import Number

import numpy as np


class Demographics(ABC):
    """Abstract class to store demographic data for agent-based models"""

    def __init__(self, nyears, ncommunities):
        self._nyears = nyears
        self._ncommunities = ncommunities
        self._population = np.zeros((nyears, ncommunities), dtype=np.uint32)
        self._births = np.zeros((nyears, ncommunities), dtype=np.uint32)
        self._deaths = np.zeros((nyears, ncommunities), dtype=np.uint32)
        self._immigrations = np.zeros((nyears, ncommunities), dtype=np.uint32)

        return

    @abstractmethod
    def initialize(self, **kwargs):
        """Initialize the demographic data - must be overridden."""

    @property
    def nyears(self):
        """Return the number of years in the demographic data."""
        return self._nyears

    @property
    def ncommunities(self):
        """Return the number of communities in the demographic data."""
        return self._ncommunities

    @property
    def population(self):
        """Return the populations of the communities in the given year."""
        return self._population

    @property
    def births(self):
        """Return the number of births in the communities in the given year."""
        return self._births

    @property
    def deaths(self):
        """Return the number of deaths in the communities in the given year."""
        return self._deaths

    @property
    def immigrations(self):
        """Return the number of immigrations into the communities in the given year."""
        return self._immigrations


class DemographicsStatic(Demographics):
    def __init__(self, nyears, ncommunities):
        super().__init__(nyears, ncommunities)
        return

    def initialize(self, **kwargs):
        """Initialize the demographic data."""
        # The following arguments should be 2 dimensional arrays of data [nyears, ncommunities]
        self._population = kwargs["population"]
        self._births = kwargs["births"]
        self._deaths = kwargs["deaths"]
        self._immigrations = kwargs["immigrations"]
        return


class DemographicsByYear(Demographics):
    def __init__(self, nyears, ncommunities):
        super().__init__(nyears, ncommunities)
        self._cbr = np.float32(0.0)
        self._mortality = np.float32(0.0)
        self._immigration = np.float32(0.0)
        return

    def initialize(self, **kwargs):
        """Initialize the demographic data."""
        # population should be ncommunities long
        self._population[0] = kwargs["initial_population"]
        # rates are per 1000, may be constant or vary by year (but not by community)
        self._cbr = kwargs["cbr"] if "cbr" in kwargs else np.float32(0.0)
        self._mortality = kwargs["mortality"] if "mortality" in kwargs else np.float32(0.0)
        self._immigration = kwargs["immigration"] if "immigration" in kwargs else np.float32(0.0)

        if isinstance(self._cbr, Number, np.number):
            self._cbr = np.full(self.nyears, self._cbr, dtype=np.float32)

        if isinstance(self._mortality, Number, np.number):
            self._mortality = np.full(self.nyears, self._mortality, dtype=np.float32)

        if isinstance(self._immigration, Number, np.number):
            self._immigration = np.full(self.nyears, self._immigration, dtype=np.float32)

        for year, cbr, mort, imm in zip(range(1, self.nyears), self._cbr, self._mortality, self._immigration):
            p = self._population[year]
            self._births[year] = np.round(p * cbr / 1000.0)
            self._deaths[year] = np.round(p * mort / 1000.0)
            self._immigrations[year] = np.round(p * imm / 1000.0)

            if year < len(self._population):
                self._population[year + 1] = p + self._births[year] - self._deaths[year] + self._immigrations[year]

        return


class DemographicsByYearStochastic(Demographics):
    def __init__(self, nyears, ncommunities):
        super().__init__(nyears, ncommunities)
        self._cbr = np.float32(0.0)
        self._mortality = np.float32(0.0)
        self._immigration = np.float32(0.0)
        return

    def initialize(self, **kwargs):
        """Initialize the demographic data."""
        # population should be ncommunities long
        self._population[0] = kwargs["initial_population"]
        # rates are per 1000, may be constant or vary by year (but not by community)
        self._cbr = kwargs["cbr"] if "cbr" in kwargs else np.float32(0.0)
        self._mortality = kwargs["mortality"] if "mortality" in kwargs else np.float32(0.0)
        self._immigration = kwargs["immigration"] if "immigration" in kwargs else np.float32(0.0)

        prng = np.random.default_rng(kwargs["seed"] if "seed" in kwargs else datetime.now(timezone.utc).milliseconds())

        if isinstance(self._cbr, Number, np.number):
            self._cbr = np.full(self.nyears, self._cbr, dtype=np.float32)

        if isinstance(self._mortality, Number, np.number):
            self._mortality = np.full(self.nyears, self._mortality, dtype=np.float32)

        if isinstance(self._immigration, Number, np.number):
            self._immigration = np.full(self.nyears, self._immigration, dtype=np.float32)

        for year, cbr, mort, imm in zip(range(1, self.nyears), self._cbr, self._mortality, self._immigration):
            p = self._population[year]
            self._births[year] = prng.poisson(np.round(p * cbr / 1000.0))
            self._deaths[year] = prng.poisson(np.round(p * mort / 1000.0))
            self._immigrations[year] = prng.poisson(np.round(p * imm / 1000.0))

            if year < len(self._population):
                self._population[year + 1] = p + self._births[year] - self._deaths[year] + self._immigrations[year]

        return
