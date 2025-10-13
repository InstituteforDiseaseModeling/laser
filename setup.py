from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"laser_core": ["*.c"]},
    include_package_data=True,
    ext_modules=[
        Extension(
            name="_extension",
            sources=["src/laser_core/_extension.c"],
            include_dirs=["include"],
            # libraries=["m"], library_dirs=[...], extra_compile_args=[...], etc.
        ),
        # more Extension(...) if you have them
    ],
)
