#!/usr/bin/env python

from setuptools import find_packages, setup


description = ("SpiNNaker backend for the Nengo neural modelling framework")
long_description = """Nengo is a suite of software used to build and simulate
large-scale brain models using the methods of the Neural Engineering Framework.
SpiNNaker is a neuromorphic hardware platform designed to run large-scale
spiking neural models in real-time. Using SpiNNaker to simulate Nengo models
allows you to run models in real-time and interface with external hardware
devices such as robots.
"""

setup(
    name="nengo_spinnaker",
    version="1.0.0-dev",
    author="CNRGlab at UWaterloo and APT Group, University of Manchester",
    author_email="https://github.com/ctn-waterloo/nengo_spinnaker/issues",
    url="https://github.com/ctn-waterloo/nengo_spinnaker",
    packages=find_packages(),
    package_data={'nengo_spinnaker': ['binaries/*.aplx']},
    scripts=[],
    license="GPLv3",
    description=description,
    long_description=long_description,
    install_requires=[
        "enum34>=1.0.3",
        "numpy>=1.6",
        "sentinel>=0.1.1",
        "six",
    ],
    requires=[
        "pacman",  # Temporary until this is moved to PyPi
        "spinnman",  # As above
        "spinnmachine",  # As above
        "nengo (==2.0.0)",  # As above
    ],
    tests_require=['pytest>=2.3', 'mock'],
    extras_require={
        'Spike probing': ['bitarray'],
        'Documentation': ['sphinx'],
    },
)
