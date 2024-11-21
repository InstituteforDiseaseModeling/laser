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
        self.load_manifest()

    def load_manifest(self):
        """Load the manifest module if it exists."""
        if os.path.isfile(self.manifest_path):
            spec = importlib.util.spec_from_file_location("manifest", self.manifest_path)
            self.manifest = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.manifest)
            print("Manifest module loaded successfully.")
        else:
            raise FileNotFoundError(f"{self.manifest_path} does not exist.")

        self.nn_nodes, self.initial_populations, self.cbrs = self.manifest.load_population_data()

    def save_pops_in_nodes(self):
        """Initialize the node populations."""
        node_count = len(self.nn_nodes)
        self.nodes = Population(capacity=node_count)
        self.nodes.add(node_count)
        self.nodes.add_vector_property("population", self.params.ticks + 1)
        self.nodes.population[0] = self.initial_populations
        self.nodes.nn_nodes = self.nn_nodes

    @staticmethod
    def propagate_population(model, tick):
        """Propagate the population to the next timestep."""
        #self.nodes.population[tick + 1] = self.nodes.population[tick]
        model.nodes.population[tick + 1] = model.nodes.population[tick]

    def init_from_data(self):
        """Initialize the model from provided data."""
        Population.create_from_capacity(self, self.initial_populations, self.cbrs)
        capacity = self.population.capacity

        from .schema import schema
        self.population.add_properties_from_schema(schema)

        self.assign_node_ids()
        self.save_pops_in_nodes()

        self.nodes.initial_infections = np.uint32(
            np.round(np.random.poisson(self.params.prevalence * self.initial_populations))
        )

        age_init.init(self, self.manifest)
        immunity.init(self)

        if self.params.viz:
            viz_pop( model )

        return capacity

    def assign_node_ids(self):
        """Assign node IDs to the population."""
        index = 0
        for nodeid, count in enumerate(self.initial_populations):
            self.population.nodeid[index : index + count] = nodeid
            index += count

    def init_from_file(self, filename):
        """Load the population from a file."""
        self.population = Population.load(filename)
        self.extend_capacity_after_loading()
        self.save_pops_in_nodes()

    def extend_capacity_after_loading(self):
        """Extend the population capacity after loading."""
        capacity = self.population.capacity
        print(f"Allocating capacity for {capacity:,} individuals")
        self.population.set_capacity(capacity)

    def check_for_cached(self):
        """Check for a cached HDF5 file and use it if available."""
        hdf5_directory = self.manifest.laser_cache
        os.makedirs(hdf5_directory, exist_ok=True)

        for filename in os.listdir(hdf5_directory):
            if filename.endswith(".h5"):
                hdf5_filepath = os.path.join(hdf5_directory, filename)
                cached = check_hdf5_attributes(
                    hdf5_filename=hdf5_filepath,
                    initial_populations=self.initial_populations,
                    age_distribution=age_init.age_distribution,
                    cumulative_deaths=cumulative_deaths,
                )
                if cached:
                    self.init_from_file(hdf5_filepath)
                    return True
        return False

