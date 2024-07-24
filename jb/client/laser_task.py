"""
IDMTools JSON configured Python task.

Copyright 2024, Bill Gates Foundation. All rights reserved.
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
    Extends CommandTask to store and pass parameters as a settings.py file.

    Attributes:
        configfile_argument (Optional[str]): Argument for the configuration file, default is "--config".

    Methods:
        __init__(command): Initializes the task with a command and loads base settings.
        __post_init__(): Post-initialization.
        _load_settings(file_path): Loads settings from a file.
        _process_settings(): Processes settings to replace values as per the configuration.
        set_parameter(name, value): Sets a parameter in the configuration.
        gather_common_assets(): Gathers common assets for the task.
        gather_transient_assets(): Gathers transient assets, primarily the settings.py file.
        reload_from_simulation(simulation, **kwargs): Reloads the task from a simulation.
        pre_creation(parent, platform): Pre-creation hook.
        post_creation(parent, platform): Post-creation hook.
    """

    configfile_argument: Optional[str] = field(default="--config")

    def __init__(self, command):
        """
        Initializes the task with a command and loads base settings.

        Args:
            command (str): Command to execute.
        """
        self.config = dict()
        self.base_settings = self._load_settings("settings.py")
        CommandTask.__init__(self, command)

    def __post_init__(self):
        """Post-initialization to call parent class's post_init."""
        CommandTask.__post_init__(self)

    def _load_settings(self, file_path):
        """
        Loads settings from a given file path.

        Args:
            file_path (str): Path to the settings file.

        Returns:
            list: List of settings lines.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        with open(file_path, 'r') as file:
            settings = file.readlines()
        return settings

    def _process_settings(self):
        """
        Processes the settings by replacing values based on the current configuration.

        Returns:
            str: Processed settings as a string.
        """
        processed_settings = []

        for line in self.base_settings:
            if '=' in line:
                key, value = map(str.strip, line.split('=', 1))
                if key in self.config:
                    value = str(self.config[key])
                    processed_settings.append(f"{key} = {value} # SET IN SCRIPT")
                else:
                    processed_settings.append(f"{key} = {value}")
            else:
                processed_settings.append(line)  # Handle lines without '='

        return "\n".join(processed_settings)

    def set_parameter(self, name: str, value: any) -> dict:
        """
        Sets a parameter in the configuration.

        Args:
            name (str): Parameter name.
            value (any): Parameter value.

        Returns:
            dict: Updated configuration.
        """
        self.config[name] = value
        return {name: value}

    def gather_common_assets(self):
        """
        Gathers common assets for the task.

        Returns:
            AssetCollection: Common assets.
        """
        return CommandTask.gather_common_assets(self)

    def gather_transient_assets(self) -> AssetCollection:
        """
        Gathers transient assets, primarily the settings.py file.

        Returns:
            AssetCollection: Transient assets.
        """
        self.transient_assets.add_or_replace_asset(Asset(filename="settings.py", content=self._process_settings()))
        return CommandTask.gather_transient_assets(self)

    def reload_from_simulation(self, simulation: Simulation, **kwargs):
        """
        Reloads the task from a simulation.

        Args:
            simulation (Simulation): Simulation to reload from.
            **kwargs: Additional arguments.
        """
        CommandTask.reload_from_simulation(self, simulation, **kwargs)

    def pre_creation(self, parent: Union[Simulation, IWorkflowItem], platform: 'IPlatform'):
        """
        Pre-creation hook.

        Args:
            parent (Union[Simulation, IWorkflowItem]): Parent of the task.
            platform ('IPlatform'): Platform the task is being executed on.
        """
        CommandTask.pre_creation(self, parent, platform)

    def post_creation(self, parent: Union[Simulation, IWorkflowItem], platform: 'IPlatform'):
        """
        Post-creation hook.

        Args:
            parent (Union[Simulation, IWorkflowItem]): Parent of the task.
            platform ('IPlatform'): Platform the task is being executed on.
        """
        CommandTask.post_creation(self, parent, platform)


class PyConfiguredSingularityTaskSpecification(TaskSpecification):
    """
    Provides the plugin information for PyConfiguredSingularityTask.

    Methods:
        get(configuration): Returns an instance of PyConfiguredSingularityTask with the given configuration.
        get_description(): Returns the description for the plugin.
        get_type(): Returns the type of the plugin.
        get_version(): Returns the version of the plugin.
    """

    def get(self, configuration: dict) -> PyConfiguredSingularityTask:
        """
        Returns an instance of PyConfiguredSingularityTask with the given configuration.

        Args:
            configuration (dict): Configuration for the task.

        Returns:
            PyConfiguredSingularityTask: Configured task instance.
        """
        return PyConfiguredSingularityTask(**configuration)

    def get_description(self) -> str:
        """
        Returns the description for the plugin.

        Returns:
            str: Plugin description.
        """
        return "Defines a Python script that has a single JSON config file"

    def get_type(self) -> Type[PyConfiguredSingularityTask]:
        """
        Returns the type for the plugin.

        Returns:
            Type[PyConfiguredSingularityTask]: Task type.
        """
        return PyConfiguredSingularityTask

    def get_version(self) -> str:
        """
        Returns the version of the plugin.

        Returns:
            str: Plugin version.
        """
        from idmtools_models import __version__
        return __version__

