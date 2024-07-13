"""idmtools json configured python task.

Copyright 2021, Bill & Melinda Gates Foundation. All rights reserved.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Type, Union, TYPE_CHECKING
from idmtools.assets import AssetCollection, Asset
from idmtools.entities.iworkflow_item import IWorkflowItem
from idmtools.entities.simulation import Simulation
from idmtools.registry.task_specification import TaskSpecification
from idmtools.entities.command_task import CommandTask

if TYPE_CHECKING:  # pragma: no cover
    from idmtools.entities.iplatform import IPlatform


@dataclass
class PyConfiguredSingularityTask(CommandTask):
    """
    PyConfiguredSingularityTask extends CommandTask to store and pass the parameters as a settings.py file.

    Notes:
        - TODO Add examples here

    See Also:
        :class:`idmtools_models.json_configured_task.JSONConfiguredTask`
        :class:`idmtools_models.python.python_task.PythonTask`
    """
    configfile_argument: Optional[str] = field(default="--config")
    def __init__(self, command):
        self.config = dict()
        self.base_settings = self._load_settings("settings.py")
        CommandTask.__init__(self, command)

    def __post_init__(self):
        """Constructor."""
        CommandTask.__post_init__(self)

    def _load_settings(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        with open(file_path, 'r') as file:
            settings = file.readlines()
        return settings

    def _process_settings(self):
        processed_settings = []

        for line in self.base_settings:
            # Split the line into key and value assuming the format is "key = value"
            if '=' in line:
                key, value = map(str.strip, line.split('=', 1))
                # Replace value if key is in self.config
                if key in self.config:
                    print( f"Setting {key}." )
                    value = str(self.config[key])
                    processed_settings.append(f"{key} = {value} # SET IN SCRIPT")
                else:
                    #print( f"Didn't find key {key} in {self.config.keys()}." )
                    processed_settings.append(f"{key} = {value}")
            else:
                processed_settings.append(line)  # Handle lines without '='

        return "\n".join(processed_settings)

    def set_parameter(self, name: str, value: any) -> dict:
        self.config[name]=value
        return {name: value}

    def gather_common_assets(self):
        """
        Return the common assets for a JSON Configured Task a derived class.

        Returns:
            Assets
        """
        return CommandTask.gather_common_assets(self)

    def gather_transient_assets(self) -> AssetCollection:
        """
        Get Transient assets. This should general be the config.json.

        Returns:
            Transient assets
        """
        # print( f"I think this is where we'd create a new settings.py out of {self.config} and assetize it." )
        self.transient_assets.add_or_replace_asset(Asset(filename="settings.py", content=self._process_settings()))
        return CommandTask.gather_transient_assets(self)

    def reload_from_simulation(self, simulation: Simulation, **kwargs):
        """
        Reload the task from a simulation.

        Args:
            simulation: Simulation to reload from
            **kwargs:

        Returns:
            None

        See Also
            :meth:`idmtools_models.json_configured_task.JSONConfiguredTask.reload_from_simulation`
            :meth:`idmtools_models.python.python_task.PythonTask.reload_from_simulation`
        """
        CommandTask.reload_from_simulation(self, simulation, **kwargs)

    def pre_creation(self, parent: Union[Simulation, IWorkflowItem], platform: 'IPlatform'):
        """
        Pre-creation.

        Args:
            parent: Parent of task
            platform: Platform Python Script is being executed on

        Returns:
            None
        See Also
            :meth:`idmtools_models.json_configured_task.JSONConfiguredTask.pre_creation`
            :meth:`idmtools_models.python.python_task.PythonTask.pre_creation`
        """
        CommandTask.pre_creation(self, parent, platform)

    def post_creation(self, parent: Union[Simulation, IWorkflowItem], platform: 'IPlatform'):
        """
        Post-creation.

        For us, we proxy the underlying JSONConfiguredTask and PythonTask/

        Args:
            parent: Parent
            platform: Platform Python Script is being executed on

        Returns:
            None

        See Also
            :meth:`idmtools_models.json_configured_task.JSONConfiguredTask.post_creation`
            :meth:`idmtools_models.python.python_task.PythonTask.post_creation`
        """
        CommandTask.post_creation(self, parent, platform)


class PyConfiguredSingularityTaskSpecification(TaskSpecification):
    """
    PyConfiguredSingularityTaskSpecification provides the plugin info for PyConfiguredSingularityTask.
    """

    def get(self, configuration: dict) -> PyConfiguredSingularityTask:
        """
        Get  instance of PyConfiguredSingularityTask with configuration.

        Args:
            configuration: Configuration for task

        Returns:
            PyConfiguredSingularityTask with configuration
        """
        return PyConfiguredSingularityTask(**configuration)

    def get_description(self) -> str:
        """
        Get description for plugin.

        Returns:
            Plugin Description
        """
        return "Defines a python script that has a single JSON config file"

    def get_type(self) -> Type[PyConfiguredSingularityTask]:
        """
        Get Type for Plugin.

        Returns:
            PyConfiguredSingularityTask
        """
        return PyConfiguredSingularityTask

    def get_version(self) -> str:
        """
        Returns the version of the plugin.

        Returns:
            Plugin Version
        """
        from idmtools_models import __version__
        return __version__
