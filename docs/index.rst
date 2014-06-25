.. Nengo SpiNNaker documentation master file, created by
   sphinx-quickstart2 on Sat May  3 10:25:11 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SpiNNaker backend for Nengo
===========================

`Nengo <https://github.com/ctn-waterloo/nengo/>`_ is a suite of software used
to build and simulate large-scale brain models using the methods of the
`Neural Engineering Framework
<http://compneuro.uwaterloo.ca/research/nef.html>`_.
`SpiNNaker <https://apt.cs.manchester.ac.uk/projects/SpiNNaker>`_ is a
neuromorphic hardware platform designed to run large-scale spiking neural
models in real-time.
Using SpiNNaker to simulate Nengo models allows you to run models
in real-time and interface with external hardware devices such as robots.

.. toctree::
   :maxdepth: 2

   installation

If you're new to Nengo we recommend reading through the Nengo documentation and
trying a few examples before progressing on to running examples on SpiNNaker.

.. toctree::
   :maxdepth: 2

   usage

Some hardware is already supported by Nengo SpiNNaker and more will be added
over time.

* `"PushBot" <https://github.com/ctn-waterloo/nengo_pushbot>`_ - `Neuroscientific System Theory (NST) <http://www.nst.ei.tum.de>`_ 
  

While Ensembles and various other components are simulated directly on the
SpiNNaker board this is, in general, not possible for Nodes, which may be any
arbitrary function.

.. toctree::
   :maxdepth: 3

   nodes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

