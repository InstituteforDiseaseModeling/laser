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

        # Identify unique node IDs and their counts
        #unique_nodes, counts = np.unique(nodeid_array, return_counts=True)

        # Initialize a population array with zeros, size max_nodeid + 1
        #node_populations = np.zeros(unique_nodes[-1] + 1, dtype=int)

        # Populate the population array with counts
        #node_populations[unique_nodes] = counts

        return node_populations 


    def eula(self, age_in_years: float):
        """Remove individuals older than age_in_years and bucketize them by age"""
        # Convert age_in_years to days
        age_threshold_in_days = int(age_in_years * 365)
        
        # Calculate the age of each individual in days
        current_day = 0  # Adjust this if you have a simulation day tracker
        ages_in_days = current_day - self.__dict__['dob']
        
        # Sort population by age
        sorted_indices = np.argsort(ages_in_days)
        sorted_ages = ages_in_days[sorted_indices]
        
        # Identify the index where ages exceed the threshold
        split_index = np.searchsorted(sorted_ages, age_threshold_in_days)
        
        # Keep only the individuals below the age threshold
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                self.__dict__[key] = value[sorted_indices[:split_index]]
        
        # Update population count
        self._count = split_index
        
        # Bucketize the removed individuals
        older_ages = sorted_ages[:split_index]
        bucket_ages = older_ages // 365  # Convert to years
        max_age = int(bucket_ages.max()) if bucket_ages.size > 0 else age_in_years
        
        # Initialize age buckets array
        self.age_buckets = np.zeros(max_age - int(age_in_years) + 1, dtype=int)
        
        # Count the number of individuals in each age bucket
        for age in range(int(age_in_years), max_age + 1):
            self.age_buckets[age - int(age_in_years)] = np.sum(bucket_ages == age)

        print( self.age_buckets ) 

    def keep_non_eula_pop_capbug( self, eula_age_in_years: float):
        """Remove individuals older than age_in_years and retain only the younger population."""
        # Convert age_in_years to days
        age_threshold_in_days = int(eula_age_in_years * 365)

        # Calculate the age of each individual in days
        current_day = 0  # Adjust this if you have a simulation day tracker
        ages_in_days = current_day - self.__dict__['dob']

        # Sort population by age
        sorted_indices = np.argsort(ages_in_days)

        # Identify the index where ages exceed the threshold
        split_index = np.searchsorted(ages_in_days[sorted_indices], age_threshold_in_days)

        # Keep only the individuals below the age threshold
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                self.__dict__[key] = value[sorted_indices[:split_index]]

        # Update population count
        self._count = split_index # capacity should have the same number of babies as before but with all the eula's removed
        self._capacity -= len(sorted_indices[split_index:])

        print(f"Population count after eula: {self._count}")

    def keep_non_eula_pop( self, eula_age_in_years: float):
        """Remove individuals older than age_in_years and retain only the younger population."""
        # Convert age_in_years to days
        age_threshold_in_days = int(eula_age_in_years * 365)

        # Calculate the age of each individual in days
        current_day = 0  # Adjust this if you have a simulation day tracker
        ages_in_days = current_day - self.__dict__['dob']

        # Sort population by age
        sorted_indices = np.argsort(ages_in_days)

        # Identify the index where ages exceed the threshold
        split_index = np.searchsorted(ages_in_days[sorted_indices], age_threshold_in_days)

        pdb.set_trace()
        # Keep only the individuals below the age threshold
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                self.__dict__[key] = value[sorted_indices[:split_index]]

        # Update population count
        self._count = split_index # capacity should have the same number of babies as before but with all the eula's removed
        self._capacity -= len(sorted_indices[split_index:])

        print(f"Population count after eula: {self._count}")


    def expected_pops_over_years_allnodes(self, eula_age_in_years=5):
        """Update total population older than eula_age_in_years with expected deaths subtracted"""
        # Get the total population count for each year
        self.total_population_per_year = np.zeros(20, dtype=int)
        eula_age_in_days = eula_age_in_years*365
        initial_mask = self.__dict__['dob'] <= -eula_age_in_days
        dod_filtered = self.__dict__['dod'][initial_mask]

        # Loop over each year from 1 to 20
        for year in range(1, 21):
            # Set the current day for this year
            current_day = year * 365

            # Count individuals older than the threshold
            total_population_count = np.sum(dod_filtered >= current_day)

            # Subtract the number of expected deaths
            self.total_population_per_year[year - 1] = total_population_count

        print( f"{self.total_population_per_year}" )

    def expected_pops_over_years_cap_bug(self, eula_age_in_years=5):
        eula_age_in_days = eula_age_in_years*365

        # Determine the initial mask for individuals older than eula_age_in_years at the start of the simulation
        initial_mask = self.__dict__['dob'] <= -eula_age_in_days
        dod_filtered = self.__dict__['dod'][initial_mask]
        nodeid_filtered = self.__dict__['nodeid'][initial_mask]

        # Initialize the dictionary to store total population counts for each nodeid per year
        unique_nodeids = np.unique(nodeid_filtered)
        #for nodeid in unique_nodeids:
        #    self.total_population_per_year[nodeid] = np.zeros(20, dtype=int)

        # Initialize the total_population_per_year array
        self.total_population_per_year = np.zeros((len(unique_nodeids), 20), dtype=int)

        for year in tqdm(range(1, 21)):
            current_day = year * 365
            for i, nodeid in enumerate(unique_nodeids):
                nodeid_mask = (nodeid_filtered == nodeid)
                total_population_count = np.sum(dod_filtered[nodeid_mask] >= current_day)
                self.total_population_per_year[i, year - 1] = total_population_count

        for nodeid, populations in enumerate(self.total_population_per_year):
            print(f"Node {nodeid}: {populations}")

    def expected_pops_over_years(self, eula_age_in_years=5):
        """Remove individuals older than age_in_years and retain only the younger population."""
        # Convert age_in_years to days
        age_threshold_in_days = int(eula_age_in_years * 365)

        # Calculate the age of each individual in days
        current_day = 0  # Adjust this if you have a simulation day tracker
        ages_in_days = current_day - self.__dict__['dob'][:self._count]

        # Sort population by age (only consider the active portion, ignoring the buffer)
        sorted_indices = np.argsort(ages_in_days)

        # Identify the index where ages exceed the threshold
        split_index = np.searchsorted(ages_in_days[sorted_indices], age_threshold_in_days)

        # Keep only the individuals below the age threshold, ignoring the buffer
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                # Create a new array with the correct size, including the buffer
                new_value = np.empty_like(value)

                # Copy the filtered population data
                new_value[:split_index] = value[sorted_indices[:split_index]]
                # Restore the buffer values
                new_value[split_index:self._capacity] = value[self._count:self._capacity]
                # Assign the new array back to the dictionary
                self.__dict__[key] = new_value

        # Update population count and adjust capacity
        self._count = split_index
        self._capacity = split_index + (self._capacity - self._count)


    def init_eula( self, eula_age_in_years=5 ):
        pdb.set_trace()
        self.expected_pops_over_years(eula_age_in_years=eula_age_in_years)
        print( "Calculated EULA population of each node for next 20 years." )
        self.keep_non_eula_pop(eula_age_in_years=eula_age_in_years)
        print( "Removed EULA agents from actively modeled population." )
        print( "self.count" )
        # update capacity
        # update count
