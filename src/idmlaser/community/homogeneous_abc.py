"""Well-mixed Agent Based Community"""

import numpy as np


class HomogeneousABC:
    """Homogeneous Agent Based Community"""
    def __init__(self, count, **kwargs):
        self.count = count
        self.steps = []
        for key, value in kwargs.items():
            setattr(self, key, value)
        return

    # dynamically add a property to the class
    def add_property(self, name, dtype=np.uint32, default=0):
        """Add a property to the class"""
        # initialize the property to a NumPy array with of size self.count, dtype, and default value
        setattr(self, name, np.full(self.count, default, dtype=dtype))
        return

    # add a processing step to be called at each time step
    def add_step(self, step):
        """Add a processing step to be called at each time step"""
        self.steps.append(step)
        return

    # run all processing steps at each time step
    def step(self, timestep: np.uint32):
        """Run all processing steps"""
        for step in self.steps:
            step(self, timestep)
        return
