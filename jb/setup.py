#!/usr/bin/env python
import re
import os
import shutil
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.install_lib import install_lib
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess

# Custom build_ext command to build the .so file
class CustomBuildExtCommand(build_ext):
    def run(self):
        self.build_shared_library()
        build_ext.run(self)

    def build_shared_library(self):
        # Path to the source file
        cpp_source = 'src/idmlaser/update_ages.cpp'
        # Path to the output shared library
        output_so = os.path.join(self.build_lib, 'idmlaser/update_ages.so')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_so), exist_ok=True)
        # Compile the shared library
        cmdline = f"g++ -shared -fPIC -O3 -march=native -flto -fpermissive -fopenmp -ffast-math {cpp_source} -o {output_so}"
        subprocess.check_call(cmdline.split())
        if self.inplace:
            dest_so = os.path.join(os.path.dirname(__file__), 'src/idmlaser/update_ages.so')
            self.copy_file(output_so, dest_so)

# Custom develop command to ensure the .so is built during develop
class CustomDevelopCommand(develop):
    def run(self):
        self.run_command('build_ext')
        develop.run(self)

# Custom install command to ensure the .so is built during install
class CustomInstallCommand(install):
    def run(self):
        self.run_command('build_ext')
        install.run(self)

# Custom build_py command to ensure the .so is included in the build
class CustomBuildCommand(build_py):
    def run(self):
        self.run_command('build_ext')
        build_py.run(self)

# Function to read files
def read(*names, **kwargs):
    with Path(__file__).parent.joinpath(*names).open(encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()

# Setup function with all your project details
setup(
    name="idmlaser",
    version="0.0.8",
    license="MIT",
    description="Light Agent Spatial modeling for ERadication.",
    long_description=re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.md")),
    long_description_content_type="text/markdown",
    author="Christopher Lorton",
    author_email="christopher.lorton@gatesfoundation.org",
    url="https://github.com/InstituteforDiseaseModeling/laser",
    packages=["idmlaser","idmlaser/utils","idmlaser/examples"],
    package_dir={"": "src"},
    py_modules=[path.stem for path in Path("src").glob("*.py")],  # Include all Python files in src as modules
    include_package_data=True,
    package_data={
        "idmlaser.model_numpy": ["*.py"],  # Include all Python files in idmlaser.model_code
        "idmlaser.model_sql": ["*.py"],  # Include all Python files in idmlaser.model_code
        "idmlaser": ["update_ages.so"],  # Include the .so file
    },
    zip_safe=False,
    project_urls={
        "Documentation": "https://laser.readthedocs.io/",
        "Changelog": "https://laser.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/InstituteforDiseaseModeling/laser/issues",
    },
    python_requires=">=3.8",
    install_requires=[
        "click",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "sparklines",
        "requests",
        "folium"
    ],
    entry_points={
        "console_scripts": [
            "idmlaser = idmlaser.cli:main",  # Ensure idmlaser/cli.py defines a main function
        ]
    },
    cmdclass={
        'build_py': CustomBuildCommand,
        'install': CustomInstallCommand,
        'build_ext': CustomBuildExtCommand,
        'develop': CustomDevelopCommand,
    },
)

