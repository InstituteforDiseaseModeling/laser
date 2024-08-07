"""Single array-based population class for agent-based models"""

from typing import Tuple

from tqdm import tqdm
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
        assert self._count + count <= self._capacity, f"New population ({self._count + count}) exceeds capacity ({self._capacity=})"
        i = self._count
        self._count += int(count)
        j = self._count
        #print( f"Adding {count} agents bringing us to {self._count}." )
        return i, j

    def __len__(self) -> int:
        return self._count

    def save_pd(self, filename: str, tail_number=0 ) -> None:
        """Save the population properties to a CSV file"""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
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

        return node_populations 


    def eliminate_eulas(self, eula_age_in_years: float):
        """
        Remove individuals older than a specified age and extend arrays for new births.

        - Convert the specified age threshold from years to days.
        - Calculate the age of each individual in days relative to the current simulation day.
        - Determine the number of expansion slots needed for new births.
        - Sort the population by age in ascending order.
        - Identify the index where individuals exceed the specified age threshold.
        - Retain only the individuals below the age threshold and extend the arrays with empty values
          to accommodate new births.
        - Update the population count to reflect the number of remaining individuals.
        """

        print( "Removing EULA agents from population. Have to age sort." )
        # Convert age_in_years to days
        age_threshold_in_days = int(eula_age_in_years * 365)
        
        # Calculate the age of each individual in days
        current_day = 0  # Adjust this if you have a simulation day tracker
        ages_in_days = current_day - self.__dict__['dob']

        # Calculate number of expansion slots
        birth_cap = (self.capacity-self.count) * 4 # hack coz right now seem to have "too many" births

        # Sort population by age
        sorted_indices = np.argsort(ages_in_days)
        sorted_ages = ages_in_days[sorted_indices]

        # Identify the index where ages exceed the threshold
        split_index = np.searchsorted(ages_in_days[sorted_indices], age_threshold_in_days)
        print( f"split_index = {split_index}" )

        # Ensure split_index does not exceed the size of the sorted_indices; probably unnecessary
        split_index = min(split_index, len(sorted_indices))

        # Keep only the individuals below the age threshold
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and value.size == self._capacity:
                try:
                    self.__dict__[key] = value[sorted_indices[:split_index]]
                    # Extend the array with "empty" values by birth_cap
                    extension = np.zeros(birth_cap, dtype=value.dtype)
                    self.__dict__[key] = np.concatenate([self.__dict__[key], extension])

                except Exception as ex:
                    raise ValueError( f"Exception resizing {key} vector." )

        # Update population count
        self._count = split_index

    def expected_pops_over_years(self, eula_age_in_years=5):
        """
        Estimate the population sizes by node for each year from 1 to 20, considering a 
        specific age threshold (eula_age_in_years). Start by filtering out individuals 
        younger than the given age at the start of the simulation. Then, calculate the 
        number of deaths for each node per year using the pre-existing data-of-death
        and use this information to compute the expected population size at each node 
        for each of the 20 years.

        TBD: Make sim length configurable
        """
        eula_age_in_days = eula_age_in_years * 365

        # Determine the initial mask for individuals older than eula_age_in_years at the start of the simulation
        initial_mask = self.__dict__['dob'] <= -eula_age_in_days
        dod_filtered = self.__dict__['dod'][initial_mask]
        nodeid_filtered = self.__dict__['nodeid'][initial_mask]

        # Calculate the year of death for each individual (0-based index for years 1-20)
        death_year = dod_filtered // 365
        death_year[death_year >= 20] = 19  # Cap deaths at year 20

        # Initialize the total_population_per_year array
        unique_nodeids = np.unique(nodeid_filtered)
        nodeid_indices = {nodeid: i for i, nodeid in enumerate(unique_nodeids)}
        self.total_population_per_year = np.zeros((len(unique_nodeids), 20), dtype=int)

        # Accumulate deaths by year and node
        for i in tqdm(range(len(death_year))):
            node_index = nodeid_indices[nodeid_filtered[i]]
            self.total_population_per_year[node_index, death_year[i]] += 1

        # Convert deaths to populations by subtracting cumulative deaths from the initial population
        initial_population_counts = np.bincount(nodeid_filtered, minlength=len(unique_nodeids))
        cumulative_deaths = np.cumsum(self.total_population_per_year, axis=1)
        self.total_population_per_year = initial_population_counts[:, None] - cumulative_deaths

        # Optional: print the resulting populations
        print(self.total_population_per_year)


    def init_eula( self, eula_age_in_years=5 ):
        # 1) Calculate the expected deaths & thus expected populations
        # in each node for each year of simulation, in the EULA cohort.
        self.expected_pops_over_years(eula_age_in_years=eula_age_in_years)
        # 2) Remove the EULA cohort from our population.
        self.eliminate_eulas(eula_age_in_years=eula_age_in_years)
