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


description = ("SpiNNaker backeng for Nengo")
setup(
    name="nengo_spinnaker",
    version="0.0.1.dev",
    author="CNRGlab at UWaterloo",
    author_email="https://github.com/tcstewar/nengo_spinnaker/issues",
    packages=['nengo_spinnaker'],
    scripts=[],
    license="GPLv3",
    description=description,
    long_description="",
    requires=[
        "nengo",
    ],
    test_suite='nengo_spinnaker.test',
)
