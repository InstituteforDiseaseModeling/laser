import os
import sys

from idmtools.assets import AssetCollection
from idmtools.core.platform_factory import Platform
from idmtools.entities import CommandLine
from idmtools.entities.command_task import CommandTask
from idmtools.builders import SimulationBuilder
from idmtools.entities.experiment import Experiment
from idmtools.entities.templated_simulation import TemplatedSimulations
from laser_task import PyConfiguredSingularityTask as PCST
from idmtools_platform_comps.utils.scheduling import add_schedule_config

#from idmtools_models.python.json_python_task import JSONConfiguredPythonTask

def update_parameter_callback(simulation, incubation_duration, base_infectivity, seasonal_multiplier, migration_fraction):
    simulation.task.set_parameter("incubation_duration", incubation_duration)
    simulation.task.set_parameter("base_infectivity", base_infectivity)
    simulation.task.set_parameter("seasonal_multiplier", seasonal_multiplier)
    simulation.task.set_parameter("migration_fraction", migration_fraction)
    ret_tags_dict = {"incubation_duration": incubation_duration, "base_infectivity": base_infectivity, "seasonal_multiplier": seasonal_multiplier, "migration_fraction": migration_fraction }
    return ret_tags_dict 


if __name__ == "__main__":
    here = os.path.dirname(__file__)

    # Create a platform to run the workitem
    #platform = Platform("CALCULON", NodeGroupName="idm_abcd", NumCores=16, NumNodes=1)
    platform = Platform("CALCULON")

    # create commandline input for the task
    cmdline = "singularity exec ./Assets/laser.sif python3 -m idmlaser.measles"
    command = CommandLine(cmdline)
    task = PCST(command=command)

    # Add our image
    task.common_assets.add_assets(AssetCollection.from_id_file("laser.id"))
    task.common_assets.add_directory('inputs_ew')
  
    ts = TemplatedSimulations(base_task=task)

    sb = SimulationBuilder()
    sb.add_multiple_parameter_sweep_definition(
            update_parameter_callback,
            incubation_duration=[6,6,8],
            base_infectivity=[3.9,4.0,4.1],
            seasonal_multiplier=[0.55, 0.60, 0.65],
            migration_fraction=[0.03, 0.04, 0.05]
        )
    
    """
    experiment = Experiment.from_task(
        task,
        name=os.path.split(sys.argv[0])[1],
        tags=dict(type='singularity', description='laser')
    )
    """
    ts.add_builder(sb)
    #add_schedule_config(ts, command=f"singularity exec ./Assets/laser.sif python3 -m idmlaser.measles", num_cores=8, NumNodes=1, node_group_name="idm_abcd")
    add_schedule_config(ts, command=f"singularity exec ./Assets/laser.sif env python3 -m idmlaser.measles", NumNodes=1, num_cores=24, node_group_name="idm_abcd", Environment={"OMP_NUM_THREADS":"24"} )
    experiment = Experiment.from_template(ts, name=os.path.split(sys.argv[0])[1])
    experiment.run(wait_until_done=True, scheduling=True)
    if experiment.succeeded:
        experiment.to_id_file("experiment.id")
