# Based on information from https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance-in-python

"""Experiment with dynamically adding properties to an existing class [instance]."""

import numpy as np


class Group:
    """A class to represent a group of individuals and their properties."""

    def __init__(self, indices):
        self.indices = indices
        return

    def __len__(self):
        first, last = self.indices[:]
        return last - first + 1 if last >= first else 0


def add_property(cls, name, field):
    """Add a computed property to a class."""

    def getter(self):
        first, last = self.indices[:]
        return field[first : last + 1]

    def setter(self, value):
        first, last = self.indices[:]
        field[first : last + 1] = value
        return

    setattr(cls, name, property(getter, setter))
    return


class Community:
    """A class to represent a community of individuals and their properties."""

    def __init__(self):
        self.indices = None
        return

    def initialize(self):
        """Initialize the community with a population of 1000 individuals."""
        # Four groups: S, E, I, and R
        # Initial populations: 990, 0, 10, 0
        self.indices = np.array([[0, 989], [990, 989], [990, 999], [999, 998]], dtype=np.uint32)
        self.susceptible = Group(self.indices[0])  # implicitly [0,:]
        self.exposed = Group(self.indices[1])
        self.infectious = Group(self.indices[2])
        self.recovered = Group(self.indices[3])
        # Three properties: dob, itimer, and susceptibility
        self.dob = np.random.randint(0, 100 * 365, 1000).astype(np.uint16)
        add_property(Group, "dob", self.dob)
        self.itimer = np.zeros(1000, dtype=np.uint8)
        add_property(Group, "itimer", self.itimer)
        self.susceptibility = np.ones(1000, dtype=np.uint8)
        add_property(Group, "susceptibility", self.susceptibility)
        return


c = Community()
c.initialize()
c.infectious.itimer[:] = np.clip(np.random.normal(5, 2, len(c.infectious)), a_min=1, a_max=None)
print(c.infectious.itimer)
...
