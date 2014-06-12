#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception, e:
        print "Forget setuptools, trying distutils..."
        from distutils.core import setup


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
    version="0.1a0",
    author="CNRGlab at UWaterloo and APT Group, University of Manchester",
    author_email="https://github.com/ctn-waterloo/nengo_spinnaker/issues",
    url="https://github.com/ctn-waterloo/nengo_spinnaker",
    packages=['nengo_spinnaker', 'nengo_spinnaker.nodes',
              'nengo_spinnaker.utils'],
    package_data={'nengo_spinnaker': ['binaries/*.aplx']},
    scripts=[],
    license="GPLv3",
    description=description,
    long_description=long_description,
    requires=[
        "nengo (>=2.0.0)",
        "numpy",
    ],
    extras_require={
        'Spike probing': ['bitarray'],
    },
    test_suite='nengo_spinnaker.test',
)
