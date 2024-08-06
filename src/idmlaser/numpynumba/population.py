"""Single array-based population class for agent-based models"""

from typing import Tuple

import numpy as np
import h5py
import pdb

class Population:
    """Array-based Agent Based Population Class"""

    def __init__(self, capacity: int, **kwargs):
        """
        Initialize a Population object.

        Args:
            capacity (int): The maximum capacity of the population (number of agents).
            **kwargs: Additional keyword arguments to set as attributes of the object.

        Returns:
            None
        """
        self._count = 0
        self._capacity = capacity
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.add_property = self.add_scalar_property  # alias
        self.node_count = -1

        return

    # dynamically add a property to the class
    def add_scalar_property(self, name: str, dtype=np.uint32, default=0) -> None:
        """Add a scalar property to the class"""
        # initialize the property to a NumPy array with of size self._count, dtype, and default value
        setattr(self, name, np.full(self._capacity, default, dtype=dtype))
        return

    def add_vector_property(self, name, length: int, dtype=np.uint32, default=0) -> None:
        """Add a vector property to the class"""
        # initialize the property to a NumPy array with of size self._count, dtype, and default value
        setattr(self, name, np.full((self._capacity, length), default, dtype=dtype))
        return

    @property
    def count(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, count: int) -> Tuple[int, int]:
        """Add agents to the population"""
        assert self._count + count <= self._capacity, f"Population exceeds capacity ({self._count=}, {count=}, {self._capacity=})"
        i = self._count
        self._count += int(count)
        j = self._count
        print( f"Adding {count} agents bringing us to {self._count}." )
        return i, j

    def __len__(self) -> int:
        return self._count

    def save_pd(self, filename: str, tail_number=0 ) -> None:
        """Save the population properties to a CSV file"""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                #data[key] = value[:self._count]  # Only save up to the current count
                if tail_number > 0:
                    print( f"Saving population of just {tail_number} agents born during sim." )
                    data[key] = value[self._count - tail_number:self._count]  # Save only the last additions elements
                else:
                    data[key] = value[:self._count]

        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Population data saved to {filename}")

    def save(self, filename: str, tail_number=0 ) -> None:
        import pdb
        pdb.set_trace()
        """Save the population properties to an HDF5 file"""
        with h5py.File(filename, 'w') as hdf:
            for key, value in self.__dict__.items():
                if isinstance(value, np.ndarray):
                    if tail_number > 0:
                        print(f"Saving population of just {tail_number} agents born during sim.")
                        data = value[self._count - tail_number:self._count]  # Save only the last tail_number elements
                    else:
                        data = value[:self._count]  # Only save up to the current count

                    # Create a dataset in the HDF5 file
                    hdf.create_dataset(key, data=data)

    def save_npz(self, filename: str, tail_number=0) -> None:
        """Save the population properties to a .npz file"""
        data_to_save = {}

        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if tail_number > 0:
                    print(f"Saving population of just {tail_number} agents born during sim.")
                    data_to_save[key] = value[self._count - tail_number:self._count]  # Save only the last tail_number elements
                else:
                    data_to_save[key] = value[:self._count]  # Only save up to the current count

        # Save to a .npz file
        np.savez_compressed(filename, **data_to_save)

    @staticmethod
    def load(filename: str) -> None:
        def load_hdf5( filename ):
            population = Population(0) # We'll do capacity automatically
            """Load the population properties from an HDF5 file"""
            with h5py.File(filename, 'r') as hdf:
                # Ensure nodeid is loaded first
                population.__dict__['nodeid'] = np.array(hdf['nodeid'])

                # Set _count to the length of the nodeid array
                population._count = len(population.__dict__['nodeid'])

                # Set node_count to the maximum value of the nodeid array
                population.node_count = np.max(population.__dict__['nodeid'])

                for key in hdf.keys():
                    # Read the dataset and assign it to the appropriate attribute
                    population.__dict__[key] = np.array(hdf[key])
                return population

        if filename.endswith( ".h5" ):
            population = load_hdf5( filename )

        # Set node_count to the maximum value of the numpy array under "nodeid"
        if 'nodeid' in population.__dict__:
            population.node_count = np.max(population.__dict__['nodeid'])+1
        else:
            raise KeyError("The 'nodeid' property is missing from the HDF5 file.")

        #population._count=len(population.nodeid)
        print( f"Loaded file with population {population._count}." )

        return population

    def set_capacity( self, new_capacity ):

        self._capacity = new_capacity
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                old_size = len(value)
                if old_size < new_capacity:
                    # Create a new array of the required size, filled with zeros (or a suitable default)
                    new_array = np.zeros(new_capacity, dtype=value.dtype)

                    # Copy the old data into the new array
                    new_array[:old_size] = value

                    # Replace the old array with the new array
                    self.__dict__[key] = new_array

        return

    def current( self ):
        # return tuple of first and last index of current cohort of interest
        return 0, self.count # not sure this is useful outside of original case

    def current_populations( self ):
        nodeid_array = self.__dict__['nodeid']

        # Use np.unique to get the counts directly
        _, counts = np.unique(nodeid_array[:self.count], return_counts=True)

        # Store counts in node_populations array
        node_populations = counts

        # Identify unique node IDs and their counts
        #unique_nodes, counts = np.unique(nodeid_array, return_counts=True)

        # Initialize a population array with zeros, size max_nodeid + 1
        #node_populations = np.zeros(unique_nodes[-1] + 1, dtype=int)

        # Populate the population array with counts
        #node_populations[unique_nodes] = counts

        return node_populations 

