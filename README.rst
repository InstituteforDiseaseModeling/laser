========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/laser/badge/?style=flat
    :target: https://laser.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/InstituteforDiseaseModeling/laser/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/InstituteforDiseaseModeling/laser/actions

.. |codecov| image:: https://codecov.io/gh/InstituteforDiseaseModeling/laser/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/InstituteforDiseaseModeling/laser

.. |version| image:: https://img.shields.io/pypi/v/idmlaser.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/idmlaser

.. |wheel| image:: https://img.shields.io/pypi/wheel/idmlaser.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/idmlaser

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/idmlaser.svg
    :alt: Supported versions
    :target: https://pypi.org/project/idmlaser

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/idmlaser.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/idmlaser

.. |commits-since| image:: https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser/v0.0.1.svg
    :alt: Commits since latest release
    :target: https://github.com/InstituteforDiseaseModeling/laser/compare/v0.0.1...main



.. end-badges

Light Agent Spatial modeling for ERadication.

* Free software: MIT license

Installation
============

::

    pip install idmlaser

You can also install the in-development version with::

    pip install https://github.com/InstituteforDiseaseModeling/laser/archive/main.zip


Documentation
=============


https://laser.readthedocs.io/


Development
===========

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
