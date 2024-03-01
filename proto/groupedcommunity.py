"""Implement a community of agents, grouped by type."""

from typing import Any

import numpy as np

_FIRST = 0
_LAST = 1


class Group:
    """A (virtual) group of agents."""

    def __init__(self, indices):
        # This is a reference (view) into the Community object's array of indices.
        # So, when we move agents, these values will get updated.
        self.indices = indices
        return

    def __len__(self):
        first, last = self.indices[:]
        return last - first + 1 if last >= first else 0


# def __add_property__(cls, name, field):
#     """Helper function to add a (virtual) property to a class."""

#     def getter(self):
#         first, last = self.indices[:]
#         return field[first : last + 1]

#     def setter(self, value):
#         first, last = self.indices[:]
#         field[first : last + 1] = value
#         return

#     setattr(cls, name, property(getter, setter))
#     return


def __add_property__(obj, name, field):
    """Helper function to add a (virtual) property to an object instance."""

    def getter(self):
        first, last = self.indices[:]
        return field[first : last + 1]

    def setter(self, value):
        first, last = self.indices[:]
        field[first : last + 1] = value
        return

    # https://stackoverflow.com/questions/2954331/dynamically-adding-property-in-python
    # properties have to go on a class, so create a new class for this object, if necessary
    cls = type(obj)
    if not hasattr(cls, "__dynamic_properties__"):
        cls = type(cls.__name__, (cls,), {"__dynamic_properties__": True})
        obj.__class__ = cls
    setattr(cls, name, property(getter, setter))

    return


class Community:
    """A community of agents."""

    def __init__(self) -> None:
        self.groupdefs = []
        self.attrdefs = []

        self._count = 0
        self.ngroups = -1
        self.igroups = None
        self.gmap = {}

        self.groups = {}
        self.attributes = []

        return

    def add_community_property(self, name: str, value: Any) -> None:
        """Add a property to the class."""
        setattr(self, name, value)
        return

    def add_agent_group(self, name: str, count: int) -> int:
        """Add a group of agents to the community."""
        index = len(self.groupdefs)
        self.groupdefs.append((name, count))
        return index

    def add_agent_property(self, name: str, dtype: type, default: int) -> None:
        """Add a property to the class."""
        self.attrdefs.append((name, dtype, default))
        return

    @property
    def count(self):
        """Return the number of agents in the community."""
        return self._count

    # def allocate(self):
    #     """Allocate memory for the agents."""
    #     self.ngroups = len(self.groupdefs)
    #     self.igroups = np.zeros((self.ngroups, 2), dtype=np.uint32)
    #     inext = 0
    #     for index, (name, count) in enumerate(self.groupdefs):
    #         self.gmap[name] = index
    #         setattr(self, f"i{name}", index)  # save on dictionary lookups
    #         self.igroups[index, _FIRST] = inext
    #         self.igroups[index, _LAST] = inext + count - 1
    #         group = Group(self.igroups[index])  # [index] is implicitly [index,:]
    #         self.groups[name] = group
    #         setattr(self, name, group)
    #         inext += count  # + 1
    #     self._count = inext
    #     for name, dtype, default in self.attrdefs:
    #         array = np.full(self.count, default, dtype=dtype)
    #         setattr(self, name, array)  # e.g. self.age
    #         self.attributes.append(array)
    #         # add a getter/setter for this property to the Group class
    #         __add_property__(Group, name, array)
    #     return

    def allocate(self):
        """Allocate memory for the agents."""
        self._count = sum(gd[1] for gd in self.groupdefs)  # get total population

        for name, dtype, default in self.attrdefs:
            array = np.full(self.count, default, dtype=dtype)
            setattr(self, name, array)  # e.g. self.age
            self.attributes.append(array)

        self.ngroups = len(self.groupdefs)
        # self.igroups = np.zeros((self.ngroups, 2), dtype=np.uint32)
        self.igroups = np.zeros((self.ngroups, 2), dtype=np.int32)

        inext = 0
        for index, (name, count) in enumerate(self.groupdefs):
            self.gmap[name] = index
            setattr(self, f"i{name}", index)  # save on dictionary lookups
            self.igroups[index, _FIRST] = inext
            self.igroups[index, _LAST] = inext + count - 1
            group = Group(self.igroups[index])  # [index] is implicitly [index,:]
            self.groups[name] = group
            setattr(self, name, group)
            inext += count  # + 1
            for (name, _, _), array in zip(self.attrdefs, self.attributes):
                # add a getter/setter for this each property to the Group instance
                __add_property__(group, name, array)

        return

    def move(self, source: int, index: int, target: int):
        """Move an agent from one group (index) to another group."""
        if target > source:
            if target == source + 1:
                isource, iswap = self.igroups[source, :]
                isource += index
                if isource != iswap:
                    for array in self.attributes:
                        array[iswap], array[isource] = array[isource], array[iswap]
                self.igroups[source, _LAST] -= 1
                self.igroups[target, _FIRST] -= 1
            else:
                for src in range(source, target):
                    dst = src + 1
                    isource, iswap = self.igroups[src, :]
                    isource += index
                    if isource != iswap:
                        for array in self.attributes:
                            array[iswap], array[isource] = array[isource], array[iswap]
                    self.igroups[src, _LAST] -= 1
                    self.igroups[dst, _FIRST] -= 1
                    # agents are moved to the beginning of the dst group
                    index = 0
        elif target < source:
            for src in range(source, target, -1):
                dst = src - 1
                iswap = self.igroups[src, _FIRST]
                isource = iswap + index
                if isource != iswap:
                    for array in self.attributes:
                        array[iswap], array[isource] = array[isource], array[iswap]
                self.igroups[src, _FIRST] += 1
                self.igroups[dst, _LAST] += 1
                # agents are moved to the end of the dst group
                index = self.igroups[dst, _LAST] - self.igroups[dst, _FIRST]

        return
