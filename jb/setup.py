#!/usr/bin/env python
import re
import os
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.install_lib import install_lib
import subprocess

class CustomBuildCommand(build_py):
    def run(self):
        build_py.run(self)

class CustomInstallCommand(install_lib):
    def run(self):
        install_lib.run(self)
        original_dir = os.getcwd()
        try:
            os.chdir(self.get_finalized_command('install').install_lib)
            cmdline = "g++ -shared -fPIC -O3 -march=native -flto -fpermissive -fopenmp -ffast-math idmlaser/update_ages.cpp -o idmlaser/update_ages.so"
            subprocess.check_call(cmdline.split())
        except Exception as ex:
            print( "Exception building update_ages.so." )
            print( str( ex ) )
        finally:
            os.chdir(original_dir)

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
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities",
    ],
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
        'install_lib': CustomInstallCommand
    },
)

