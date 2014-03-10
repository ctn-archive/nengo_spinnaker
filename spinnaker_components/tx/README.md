SpiNNaker/Nengo Integration - Transmit Component
================================================

Authors:
  * Andrew Mundy -- [University of Manchester](http://spinnaker.cs.man.ac.uk)
  * Terry Stewart -- [University of Waterloo](http://nengo.ca)

Introduction
------------

The Transmit (Tx) component exists to relay decoded values from a SpiNNaker
board to the host computer.  This component is automatically added when it is
necessary to perform computation upon the host.


Operation
---------

Each Tx component is expected to handle the reporting of a subset of a model's
Ensembles.  On receiving a multicast (MC) packet the Tx component uses the key
to determine which dimension of which decoder is being broadcast, the
accumulator for this dimension is updated.  On each timer tick (typically once
per simulation time step) a packet is dispatched to the host containing the
latest held values.

The receiver component on the host is reponsible for translating the decoder
and dimension number related by the Tx component into inputs to the appropriate
Node.
