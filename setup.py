import re
from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with Path(__file__).parent.joinpath(*names).open(encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()


def get_version():
    version_file = Path("src/laser_core/__init__.py")
    version_match = re.search(r'^__version__ = ["\']([^"\']+)["\']', version_file.read_text(encoding="utf8"), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="laser_core",  # Specify the name of your package
    version=get_version(),  # Fetch version from __init__.py
    description="A description of the laser_core package",  # Add a description for the package
    long_description="{}\n{}".format(
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.rst")),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    long_description_content_type="text/x-rst",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[path.stem for path in Path("src").glob("*.py")],
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
)
