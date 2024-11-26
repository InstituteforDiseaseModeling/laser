"""Single array-based population class for agent-based models."""

from typing import Tuple

from tqdm import tqdm
import numpy as np
import numba as nb
import h5py
import os
import pdb

@nb.njit(parallel=True)
def accumulate_deaths_parallel(nodeid_filtered, death_year, deaths_per_year):
    for i in nb.prange(len(death_year)):
        #node_index = nodeid_indices_array[nodeid_filtered[i]]
        node_index = nodeid_filtered[i]
        deaths_per_year[node_index, death_year[i]] += 1

def check_hdf5_attributes(hdf5_filename, initial_populations, age_distribution, cumulative_deaths, eula_age=None):
    """Verify that attributes stored in an HDF5 file match the provided data.

    This function checks whether the specified attributes in an HDF5 file correspond
    to the given input arrays. If the file or attributes are missing, or if any of
    the attributes do not match, it provides warnings or debug output.

    Parameters:
    hdf5_filename (str): The path to the HDF5 file.
    initial_populations (numpy.ndarray): Expected array for the 'init_pops' attribute.
    age_distribution (numpy.ndarray): Expected array for the 'age_dist' attribute.
    cumulative_deaths (numpy.ndarray): Expected array for the 'cumulative_deaths' attribute.
    eula_age (optional, Any): Expected value for the 'eula_age' attribute. If not provided,
                              this attribute will be ignored during comparison.

    Returns:
    bool: True if all attributes in the file match the provided data, False otherwise.

    Notes:
    - If the file does not exist, a warning is printed, and the function returns False.
    - Debugging information is printed when attributes do not match, showing which
      comparisons failed.
    - The function handles missing attributes gracefully, raising a KeyError with a
      descriptive error message.

    Example:
    >>> success = check_hdf5_attributes(
    ...     "data.h5",
    ...     np.array([100, 200]),
    ...     np.array([0.2, 0.8]),
    ...     np.array([10, 20]),
    ...     eula_age=65
    ... )
    >>> if success:
    ...     print("Attributes match.")
    ... else:
    ...     print("Attributes do not match.")
    """
    if not os.path.exists( hdf5_filename ):
        print( "WARNING: Couldn't find requested file: {hdf5_filename}" )
        return False
    with h5py.File(hdf5_filename, 'r') as hdf:
        try:
            # Retrieve the attributes from the file
            file_initial_populations = hdf.attrs['init_pops']
            file_age_distribution = hdf.attrs['age_dist']
            file_cumulative_deaths = hdf.attrs['cumulative_deaths']
            file_eula_age = hdf.attrs.get('eula_age', None)
            #print( f"file_eula_age={file_eula_age}" )

            # Compare the attributes
            if (np.array_equal(initial_populations, file_initial_populations) and
                np.array_equal(age_distribution, file_age_distribution) and
                np.array_equal(cumulative_deaths, file_cumulative_deaths) and
                eula_age == file_eula_age):
                return True
            else:
                print( "DEBUG: No match." )
                print( f"init_pop? {np.array_equal(initial_populations, file_initial_populations)} " )
                print( f"age_dist? {np.array_equal(age_distribution, file_age_distribution)} " )
                print( f"cum_death? {np.array_equal(cumulative_deaths, file_cumulative_deaths)} " )
                print( f"eula_age? {np.array_equal(eula_age, file_eula_age)} " )
                return False
        except KeyError as e:
            print(f"Attribute not found in file {hdf5_filename}\nError: {e}")
            return False


from laser_core.laserframe import LaserFrame

class ExtendedLF(LaserFrame):
    def __init__(self, capacity, **kwargs):
        # Initialize the parent class
        self.expected_new_deaths_per_year = None
        super().__init__(capacity, **kwargs)

    # Add scalar properties to model.population
    def add_properties_from_schema( self, schema ):
        """Initialize scalar properties for agents in the population based on a
        provided schema.

        This method iterates through each property defined in the schema and calls
        `add_scalar_property` to add the property to the model's population. The schema is
        expected to be a dictionary where:
            - Keys are property names (as strings).
            - Values are the corresponding data types (e.g., int, float, str).

        Purpose:
        This method provides a declarative approach to initializing agent properties based
        on a schema. It is optional and complements programmatic initialization.

        Parameters:
        schema (dict): A dictionary defining agent properties. Each key represents a property
                       name, and its value represents the property's data type.

        Example:
        schema = {
            "age": int,
            "income": float,
            "employed": bool
        }
        model.add_properties_from_schema(schema)

        After the method is called, the population will have the `age`, `income`, and `employed`
        properties available.
        """
        for name, dtype in schema.items():
            self.add_scalar_property(name, dtype)

    @staticmethod

    def create_from_capacity(model, initial_populations, cbrs=None):
        """
        Allocate and initialize a population model based on an initial population
        and projected growth capacity.

        This function calculates the required population capacity for a model by
        summing the initial populations and applying growth projections. Growth
        is determined either from crude birth rates (CBRs) provided as input or
        derived from the model's parameters. A buffer is added to the calculated
        capacity to ensure sufficient allocation.

        Parameters
        ----------
        model : object
            The model to which the population will be assigned. Must contain a
            `params` attribute with `cbr` (crude birth rate) and `ticks` (simulation
            duration in days).
        initial_populations : numpy.ndarray
            Array of initial population counts.
        cbrs : dict, optional
            A dictionary mapping regions or groups to crude birth rates (per 1000
            individuals). If provided, these values are used for growth calculations.

        Returns
        -------
        int
            The calculated population capacity after accounting for growth and a
            1% buffer.

        Notes
        -----
        - Growth is compounded annually based on crude birth rates.
        - If `cbrs` is not provided, the model's global `cbr` value is used.
        - The capacity is converted to a 32-bit unsigned integer to optimize memory usage.

        Examples
        --------
        >>> model.params.cbr = 25  # crude birth rate (per 1000 individuals)
        >>> model.params.ticks = 730  # 2 years in simulation days
        >>> initial_populations = np.array([1000, 2000, 1500])
        >>> capacity = create_from_capacity(model, initial_populations)
        >>> print(f"Allocated capacity: {capacity}")
        """
    
        # Calculate initial capacity based on sum of initial populations
        capacity = initial_populations.sum()
        print(f"initial {capacity=:,}")
        
        # Calculate growth based on cbrs or model parameters
        if cbrs:
            print(f"{cbrs=}, {model.params.ticks=}")    # type: ignore
            growth = ((1.0 + np.mean(np.array(list(cbrs.values())) / 1000)) 
                      ** (model.params.ticks // 365))
        else:
            print(f"{model.params.cbr=}, {model.params.ticks=}")    # type: ignore
            growth = ((1.0 + model.params.cbr / 1000) 
                      ** (model.params.ticks // 365))   # type: ignore
        
        print(f"{growth=}")
        
        # Adjust capacity by growth and add a 1% buffer
        capacity *= growth
        capacity *= 1.01  # 1% buffer
        capacity = np.uint32(np.round(capacity))
        
        print(f"required {capacity=:,}")
        print(f"Allocating capacity for {capacity:,} individuals")
        
        # Initialize the Population object with calculated capacity
        population = ExtendedLF(capacity)
        model.population = population   # type: ignore
        
        # Add initial population to the model's population and return capacity
        ifirst, ilast = population.add(initial_populations.sum())
        print(f"{ifirst=:,}, {ilast=:,}")
        
        return capacity

    # Added as special case of add_vector_property but probably not needed
    def add_report_property(self, name, length: int, dtype=np.uint32, default=0) -> None:
        """Add a vector property to the class."""
        # initialize the property to a NumPy array with of size self._count, dtype, and default value
        setattr(self, name, np.full((length, self._capacity), default, dtype=dtype))
        return

    def save_pd(self, filename: str, tail_number=0 ) -> None:
        """Save the population properties to a CSV file.

        Much slower than save which uses HDF5. Only use if you really
        don't want to convert your HDF5 to CSV afterwards and are
        willing to accept slowness. If you're running with smaller
        populations this might be fine.
        """
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

    def save_npz(self, filename: str, tail_number=0) -> None:
        """Save the population properties to a .npz file.

        Slower than save which uses HDF5.
        """
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

    def save(self, filename: str, tail_number=0, initial_populations=None, age_distribution=None, cumulative_deaths=None, eula_age=None ) -> None:
        """Save the population properties to an HDF5 file."""
        with h5py.File(filename, 'w') as hdf:
            hdf.attrs['count'] = self._count
            hdf.attrs['capacity'] = self._capacity
            hdf.attrs['node_count'] = len(initial_populations)
            print( "TBD: Need to derive node count since we don't have it here." )
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

    @staticmethod
    def load(filename: str) -> None:
        """Load a serialized population from a file and reconstruct its
        properties.

        This method reads population data from a specified file and recreates an
        `ExtendedLF` object. The data is expected to be stored in an HDF5 format,
        containing population properties and metadata.

        Parameters:
        filename (str): The path to the file containing the serialized population data.
                        The file must have an ".h5" extension.

        Returns:
        ExtendedLF: An instance of `ExtendedLF` initialized with the loaded data.

        Raises:
        KeyError: If the file is missing the required `nodeid` attribute.
        ValueError: If the file format is unsupported.

        Notes:
        - The file must contain at least the following attributes: `count`, `capacity`,
          `node_count`, and a dataset named `nodeid`.
        - Additional datasets within the HDF5 file are dynamically assigned to the
          `ExtendedLF` object as attributes.
        - This method automatically calculates `node_count` based on the maximum
          value in the `nodeid` array.

        Example:
        >>> population = ExtendedLF.load("population_data.h5")
        >>> print(population._count)
        10000

        Implementation Details:
        - The method uses a helper function `load_hdf5` to handle file-specific
          operations.
        - The `population.node_count` is incremented by 1 to represent the total
          number of unique nodes (assumes 0-based indexing).
        """
        def load_hdf5( filename ):
            population = ExtendedLF(0) # We'll do capacity automatically
            """Load the population properties from an HDF5 file."""
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

        print( f"Loaded file with population {population._count}." )

        return population

    def set_capacity( self, new_capacity ):
        """Adjusts the capacity of the population by resizing all the
        attributes (NumPy arrays).

        This method updates the population's capacity by checking all of its attributes stored in `__dict__`.
        If an attribute is a 1D NumPy array and its length is smaller than the new capacity, the array is
        resized to the new capacity with zeros filling the extra space. The old data is retained, and the
        resized array replaces the original one in the object.

        Attributes with 2D arrays are ignored, and a message is printed for each ignored attribute. These are
        assumed to not be population properties.

        Parameters:
        ----------
        new_capacity : int
            The new capacity to be set for the population's 1D array attributes. All 1D arrays will be resized
            to this capacity, and their contents will be preserved up to the size of the original array.

        Returns:
        -------
        None
            This method modifies the population in place, resizing arrays where necessary, and does not return anything.

        Notes:
        -----
        - Any exceptions encountered during the resizing process (e.g., copying data into the new array) will
          be caught and printed for debugging purposes, with an optional breakpoint (`pdb.set_trace()`) for
          interactive debugging.
        - 2D arrays are not resized and will be ignored during the process.

        Example:
        --------
        population.set_capacity(1000)  # Resizes all 1D arrays in the object to a length of 1000.
        """

        self._capacity = new_capacity
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                # Ignore 2D arrays
                if value.ndim == 2:
                    print( f"Ignoring key {key} while expanding capacity." )
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

    # potentially obsolete functions below here.

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


    def eliminate_eulas(self, split_index: int):
        """
        Remove individuals older than a specified age but keep extend arrays for new births.
        - Retain only the individuals below the age threshold and extend the arrays with empty values
          to accommodate new births.
        - Update the population count, and cap, to reflect the number of remaining individuals.
        """

        print( "**\nRemoving EULA agents from population. Assumes already age sorted.\n**" )

        # Keep only the individuals below the age threshold
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and value.size >= self._count: # for some reason the value to check against here has been really annoying
                try:
                    self.__dict__[key] = value[split_index:]
                    print( f"Cropping array {key}." )

                except Exception as ex:
                    raise ValueError( f"Exception resizing {key} vector." )

        # Update population count
        self._count -= split_index
        self._capacity -= split_index 
        print( f"After EULA elimination using split index {split_index}, count={self.count} and capacity={self.capacity}" )


    def expected_deaths_over_sim(self, death_years, split_index, sim_years=10):
        """Estimate the population sizes by node for each year from 1 to years,
        considering a specific age threshold (eula_age_in_years). Start by
        filtering out individuals younger than the given age at the start of
        the simulation. Then, calculate the number of deaths for each node per
        year using the pre-existing data-of-death and use this information to
        compute the expected population size at each node for each of the
        years.

        Assume nodeid but nothing else.
        """
        # Let's initially count for all the years, even if we only keep first 10.
        # No, too slow. Let's count 0 through years-1, and put 'everything else' in 'years'
        death_years[death_years > sim_years] = sim_years  # Cap deaths at years-1

        # Initialize the expected_new_deaths array
        nodeids_filtered = self.nodeid[0:split_index]
        unique_nodeids = np.unique(nodeids_filtered) # slow way of calculating node count
        nodeid_indices = {nodeid: i for i, nodeid in enumerate(unique_nodeids)}
        self.expected_new_deaths_per_year = np.zeros((len(unique_nodeids), sim_years+1), dtype=int)

        accumulate_deaths_parallel(nodeids_filtered, death_years, self.expected_new_deaths_per_year)

        # Convert deaths to populations by subtracting cumulative deaths from the initial population
        cumulative_deaths = np.cumsum(self.expected_new_deaths_per_year, axis=1) # xtra dupe element at end I don't understand yet
        
        # Calculate new deaths per year
        self.expected_new_deaths_per_year[:, 0] = self.expected_new_deaths_per_year[:, 0]
        self.expected_new_deaths_per_year[:, 1:] = np.diff(cumulative_deaths, axis=1)
        #self.expected_new_deaths_per_year = self.expected_new_deaths_per_year[:, 0:10] # discard extra killall column

        # Calculate the initial population counts, though not currently used
        initial_population_counts = np.bincount(nodeids_filtered, minlength=len(unique_nodeids))
        # Calculate total_population_per_year at the end, using cumulative deaths
        self.total_population_per_year = initial_population_counts[:, None] - cumulative_deaths

        # Optional: print the resulting populations
        print(self.total_population_per_year)

