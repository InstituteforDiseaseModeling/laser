"""Single array-based population class for agent-based models"""

from typing import Tuple

from tqdm import tqdm
import numpy as np
import numba as nb
import h5py
import pdb

@nb.njit(parallel=True)
def accumulate_deaths_parallel(nodeid_filtered, death_year, nodeid_indices_array, total_population_per_year):
    for i in nb.prange(len(death_year)):
        node_index = nodeid_indices_array[nodeid_filtered[i]]
        total_population_per_year[node_index, death_year[i]] += 1

def check_hdf5_attributes(hdf5_filename, initial_populations, age_distribution, cumulative_deaths, eula_age):
    with h5py.File(hdf5_filename, 'r') as hdf:
        try:
            # Retrieve the attributes from the file
            file_initial_populations = hdf.attrs['init_pops']
            file_age_distribution = hdf.attrs['age_dist']
            file_cumulative_deaths = hdf.attrs['cumulative_deaths']
            file_eula_age = hdf.attrs['eula_age']

            # Compare the attributes
            if (np.array_equal(initial_populations, file_initial_populations) and
                np.array_equal(age_distribution, file_age_distribution) and
                np.array_equal(cumulative_deaths, file_cumulative_deaths) and
                np.array_equal(eula_age, file_eula_age)):
                return True
            else:
                return False
        except KeyError as e:
            print(f"Attribute not found in file: {e}")
            return False

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

    def save(self, filename: str, tail_number=0, initial_populations=None, age_distribution=None, cumulative_deaths=None, eula_age=None ) -> None:
        """Save the population properties to an HDF5 file"""
        with h5py.File(filename, 'w') as hdf:
            hdf.attrs['count'] = self._count
            hdf.attrs['capacity'] = self._capacity
            hdf.attrs['node_count'] = self.node_count
            if initial_populations is not None:
                hdf.attrs['init_pops'] = initial_populations
            if age_distribution is not None:
                hdf.attrs['age_dist'] = age_distribution
            if cumulative_deaths is not None:
                hdf.attrs['cumulative_deaths'] = cumulative_deaths
            if eula_age is not None:
                hdf.attrs['eula_age'] = eula_age

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
                population._count = hdf.attrs['count']
                population._capacity = hdf.attrs['capacity']
                population.node_count = hdf.attrs['node_count']
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
                # Ignore 2D arrays
                if value.ndim == 2:
                    print( "Ignoring key {key} while expanding capacity." )
                    continue

                old_size = len(value)
                if old_size < new_capacity:
                    # Create a new array of the required size, filled with zeros (or a suitable default)
                    new_array = np.zeros(new_capacity, dtype=value.dtype)
                    try:
                        # Copy the old data into the new array
                        new_array[:old_size] = value

                        # Replace the old array with the new array
                        self.__dict__[key] = new_array
                    except Exception as ex:
                        print( str( ex ) )
                        pdb.set_trace()

        return

    def current( self ):
        # return tuple of first and last index of current cohort of interest
        return 0, self.count # not sure this is useful outside of original case

    def current_populations( self ):
        print( "NOTE: current_populations fn actually implemented as initial_populations." );
        # TBD: maybe initial is all we actually need?
        nodeid_array = self.__dict__['nodeid']

        # Use np.unique to get the counts directly
        _, counts = np.unique(nodeid_array[:self.count], return_counts=True)
        counts += self.total_population_per_year[:,0] # why 0? Need year param

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
        ages_in_days = current_day - self.__dict__['dob'][:self.count]

        # Calculate number of expansion slots
        birth_cap = (self.capacity-self.count)

        # Sort population by age
        sorted_indices = np.argsort(ages_in_days)
        sorted_ages = ages_in_days[sorted_indices]

        # Identify the index where ages exceed the threshold
        split_index = np.searchsorted(ages_in_days[sorted_indices], age_threshold_in_days)
        print( f"split_index = {split_index}" )

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

    def expected_pops_over_years(self, eula_age_in_years=5, years=10):
        """
        Estimate the population sizes by node for each year from 1 to years, considering a 
        specific age threshold (eula_age_in_years). Start by filtering out individuals 
        younger than the given age at the start of the simulation. Then, calculate the 
        number of deaths for each node per year using the pre-existing data-of-death
        and use this information to compute the expected population size at each node 
        for each of the years.

        TBD: Make sim length configurable
        """
        eula_age_in_days = eula_age_in_years * 365
        years += 1

        # Determine the initial mask for individuals older than eula_age_in_years at the start of the simulation
        initial_mask = self.__dict__['dob'] <= -eula_age_in_days
        dod_filtered = self.__dict__['dod'][initial_mask]
        nodeid_filtered = self.__dict__['nodeid'][initial_mask]

        # Calculate the year of death for each individual (0-based index for years 1-years)
        death_year = dod_filtered // 365
        # Let's initially count for all the years, even if we only keep first 10.
        # No, too slow. Let's count 0 through years-1, and put 'everything else' in 'years'
        death_year[death_year >= years] = years-1  # Cap deaths at years-1

        # Initialize the expected_new_deaths array
        unique_nodeids = np.unique(nodeid_filtered)
        nodeid_indices = {nodeid: i for i, nodeid in enumerate(unique_nodeids)}
        self.expected_new_deaths_per_year = np.zeros((len(unique_nodeids), years), dtype=int)

        nodeid_indices_array = np.zeros(unique_nodeids.max() + 1, dtype=np.int32)
        for i, nodeid in enumerate(unique_nodeids):
            nodeid_indices_array[nodeid] = i
        accumulate_deaths_parallel(nodeid_filtered, death_year, nodeid_indices_array, self.expected_new_deaths_per_year)

        # Convert deaths to populations by subtracting cumulative deaths from the initial population
        cumulative_deaths = np.cumsum(self.expected_new_deaths_per_year, axis=1) # xtra dupe element at end I don't understand yet
        
        # Calculate new deaths per year
        self.expected_new_deaths_per_year[:, 0] = self.expected_new_deaths_per_year[:, 0]
        self.expected_new_deaths_per_year[:, 1:] = np.diff(cumulative_deaths, axis=1)
        self.expected_new_deaths_per_year = self.expected_new_deaths_per_year[:, 0:10] # discard extra killall column

        # Calculate the initial population counts, though not currently used
        initial_population_counts = np.bincount(nodeid_filtered, minlength=len(unique_nodeids))
        # Calculate total_population_per_year at the end, using cumulative deaths
        self.total_population_per_year = initial_population_counts[:, None] - cumulative_deaths

        # Optional: print the resulting populations
        print(self.total_population_per_year)


    def init_eula( self, eula_age_in_years=5 ):
        # 1) Calculate the expected deaths & thus expected populations
        # in each node for each year of simulation, in the EULA cohort.
        self.expected_pops_over_years(eula_age_in_years=eula_age_in_years)
        # 2) Remove the EULA cohort from our population.
        self.eliminate_eulas(eula_age_in_years=eula_age_in_years)
