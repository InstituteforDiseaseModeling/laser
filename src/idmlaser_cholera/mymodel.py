import os
import importlib.util
import numpy as np
from idmlaser_cholera.numpynumba.population import ExtendedLF as Population
from idmlaser_cholera.mods import immunity, age_init
from idmlaser_cholera.numpynumba.population import check_hdf5_attributes
from idmlaser_cholera.demographics import cumulative_deaths
from idmlaser_cholera.utils import viz_2D, viz_pop
import pdb


class Model:
    def __init__(self, params):
        self.params = params
        self.nn_nodes = None
        self.initial_populations = None
        self.cbrs = None
        self.nodes = None
        self.population = None

        print(f"Input directory set to: {self.params.input_dir}")
        self.manifest_path = os.path.join(self.params.input_dir, "manifest.py")
        self.manifest = None
        self._load_manifest()
        age_init.age_data_manager.set_path( self.manifest.age_data )

    @classmethod
    def get(cls, params):
        """Factory method to create and initialize a Model instance.

        Handles cached data or initializes from data as needed.
        """
        model = cls(params)  # Create a Model instance
        if model._check_for_cached():
            print("*\nFound cached file. Using it.\n*")
        else:
            model._init_from_data()  # Initialize from data if no cache found
        return model

    def save( self, filename ):
        """
        Save the full agent population state to disk in HDF5 format.

        This method stores the agent population, including the initial population
        distribution across patches, age structure, and natural mortality profile,
        in a specified HDF5 file. The data stored includes:

        - `initial_populations`: The initial population count for each patch in the model.
        - `age_distribution`: The distribution of agents across age groups at the time of saving.
        - `cumulative_deaths`: The cumulative number of deaths recorded in the simulation up to this point.

        Before saving, the method checks that the `age_distribution` data has been initialized.
        If the `age_distribution` is not properly set, a `ValueError` is raised.

        Args:
            filename (str): The path to the HDF5 file where the population data should be saved.

        Raises:
            ValueError: If the `age_distribution` is not initialized before attempting to save.

        Notes:
            - The method utilizes an external `age_data_manager` to retrieve the current age distribution.
            - The `population.save` function is responsible for handling the actual file-saving process,
              including packaging the data into an HDF5 file format.
            - This method assumes that the agent population and the other components (age distribution,
              mortality data) are properly initialized before being saved.

        Example:
            # Save the population data to 'population_data.h5'
            model.save('population_data.h5')
        """
        if age_init.age_data_manager.get_data() is None:
            raise ValueError( f"age_distribution uninitialized while saving" )
        self.population.save( filename=filename,
            initial_populations=self.initial_populations,
            age_distribution=age_init.age_data_manager.get_data(),
            cumulative_deaths=cumulative_deaths
        )
    def _load_manifest(self):
        """Load the manifest module if it exists."""
        if os.path.isfile(self.manifest_path):
            spec = importlib.util.spec_from_file_location("manifest", self.manifest_path)
            self.manifest = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.manifest)
            print("Manifest module loaded successfully.")
        else:
            raise FileNotFoundError(f"{self.manifest_path} does not exist.")

        self.nn_nodes, self.initial_populations, self.cbrs = self.manifest.load_population_data()
        def convert_to_laserframe():
            # Let's convert the input data into a laserframe
            # with init_pop, cbr, lat, long, and id for each node
            # The laserframe doesn't get used yet
            self.input_pop = Population(capacity=len(self.nn_nodes))
            self.input_pop.add_scalar_property(name="populations")
            self.input_pop.add_scalar_property(name="cbrs")
            self.input_pop.add_scalar_property(name="ids")
            # Define new scalar properties for 'name', 'lat', and 'long'
            #self.input_pop.add_scalar_property(name='name', dtype=np.object)  # Store names as objects (strings)
            self.input_pop.add_scalar_property(name='lat', dtype=np.float32)  # Latitude as float
            self.input_pop.add_scalar_property(name='long', dtype=np.float32) # Longitude as float

            # print( f"{self.nn_nodes=}" )
            self.input_pop.ids[:] = np.arange(len(self.nn_nodes))
            self.input_pop.populations[:] = self.initial_populations
            self.input_pop.cbrs[:] = np.array(list(self.cbrs.values()), dtype=np.float32)

            # Populate the scalar properties from `self.nn_nodes`
            names = []
            lats = []
            longs = []

            for node, data in self.nn_nodes.items():
                names.append(node)                      # Extract the node name
                lats.append(data[1][0])                 # Extract latitude from the second tuple
                longs.append(data[1][1])                # Extract longitude from the second tuple

            # Assign the extracted data to the scalar properties
            #self.input_pop.name[:] = np.array(names, dtype=np.object)
            self.input_pop.lat[:] = np.array(lats, dtype=np.float32)
            self.input_pop.long[:] = np.array(longs, dtype=np.float32)
        convert_to_laserframe()

    def _save_pops_in_nodes(self):
        """Initialize the node populations."""
        node_count = len(self.nn_nodes)
        self.nodes = Population(capacity=node_count, initial_pop=node_count)
        #self.nodes.add(node_count)
        self.nodes.add_vector_property("population", self.params.ticks + 1)
        self.nodes.population[0] = self.initial_populations
        self.nodes.nn_nodes = self.nn_nodes

    @staticmethod
    def propagate_population(model, tick):
        """Propagate the population to the next timestep."""
        #self.nodes.population[tick + 1] = self.nodes.population[tick]
        model.nodes.population[tick + 1] = model.nodes.population[tick]

    def _init_from_data(self):
        """Initialize the model from provided data."""
        Population.create_from_capacity(self, self.initial_populations, self.cbrs)
        capacity = self.population.capacity

        from .schema import schema
        self.population.add_properties_from_schema(schema)

        self._assign_node_ids()
        self._save_pops_in_nodes()

        self.nodes.initial_infections = np.uint32(
            np.round(np.random.poisson(self.params.prevalence * self.initial_populations))
        )
        self.nodes.initial_infections[1:] = 0
        
        # It would be cleaner if these didn't need to be here
        age_init.init(self, self.manifest)
        immunity.init(self)

        if self.params.viz:
            viz_pop( model )

        return capacity

    def _assign_node_ids(self):
        """Assign node IDs to the population."""
        index = 0
        for nodeid, count in enumerate(self.initial_populations):
            self.population.nodeid[index : index + count] = nodeid
            index += count

    def _init_from_file(self, filename):
        """Load the population from a file."""
        self.population = Population.load(filename)
        self._extend_capacity_after_loading()
        self._save_pops_in_nodes()
        # We aren't yet storing the initial infections in the cached file so recreating on reload
        self.nodes.initial_infections = np.uint32(
            np.round(np.random.poisson(self.params.prevalence * self.initial_populations))
        )

    def _extend_capacity_after_loading(self):
        """Extend the population capacity after loading."""
        capacity = self.population.capacity
        print(f"Allocating capacity for {capacity:,} individuals")
        self.population.set_capacity(capacity)

    def _check_for_cached(self):
        """Check for a cached HDF5 file and use it if available."""
        hdf5_directory = self.manifest.laser_cache
        os.makedirs(hdf5_directory, exist_ok=True)

        for filename in os.listdir(hdf5_directory):
            if filename.endswith(".h5"):
                hdf5_filepath = os.path.join(hdf5_directory, filename)
                if age_init.age_data_manager.get_data() is None:
                    raise RuntimeError( "age_init.age_distribution seems to None while caching" )
                cached = check_hdf5_attributes(
                    hdf5_filename=hdf5_filepath,
                    initial_populations=self.initial_populations,
                    age_distribution=age_init.age_data_manager.get_data(),
                    cumulative_deaths=cumulative_deaths,
                )
                if cached:
                    self._init_from_file(hdf5_filepath)
                    return True
        return False

