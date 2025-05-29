============
Installation
============

LASER is currently composed of the following packages:
`laser <https://github.com/InstituteforDiseaseModeling/laser>`_,
`laser-generic <https://github.com/InstituteforDiseaseModeling/laser-generic>`_,
and `laser-measles <https://github.com/InstituteforDiseaseModeling/laser-measles>`_.

Laser contains the core components for using LASER and would most likely only need to be
installed by software engineers that would like to extend the functionality. Laser-generic
contains SIR models. Laser-measles is for the disease modeling of measles. All packages can
be installed using pip, as shown in the following example for laser-generic.

At the command line::

    pip install laser-generic

You can also install the in-development version with::

    pip install https://github.com/InstituteforDiseaseModeling/laser/archive/main.zip

Development
===========

To use LASER in a project::

    import laser_core

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox