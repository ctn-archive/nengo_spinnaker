============
Installation
============

Requirements
============

Nengo SpiNNaker requires that you have installed appropriate versions of
`Nengo <https://github.com/ctn-waterloo/nengo>`_ and the `SpiNNaker tools
package <https://spinnaker.cs.manchester.ac.uk>`_.

Basic Installation
==================

We're working towards providing the Nengo SpiNNaker package on PyPi at which
point you will be able to::

  pip install nengo_spinnaker

For now, like Nengo itself, do a developer installation.

Developer Installation
======================

If you plan to make changes to Nengo SpiNNaker you should clone its git
repository and install from it::

  git clone https://github.com/ctn-waterloo/nengo_spinnaker
  cd nengo_spinnaker
  python setup.py develop --user

If you're in a virtualenv you can omit the ``--user`` flag.

Building the SpiNNaker binaries
-------------------------------

If you installed the Nengo SpiNNaker package from source you will need to go
through a few additional steps prior to running Nengo models.  These steps
build the executable binaries which are loaded to the SpiNNaker machine. ::

  # Change to the root directory of the SpiNNaker package
  # Edit `spinnaker_tools/setup` to point at your ARM cross-compilers
  source ./setup
  make

  # Now move to the root directory of the Nengo SpiNNaker package
  cd spinnaker_components
  make
