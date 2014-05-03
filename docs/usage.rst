=================================
Running Nengo models on SpiNNaker
=================================

If this is how your Nengo model currently works::

  import nengo

  model = nengo.Network()
  with model:
      # ...

  sim = nengo.Simulator(model)
  sim.run(10.)

Then porting it to Nengo SpiNNaker requires very few changes::

  import nengo
  import nengo_spinnaker

  model = nengo.Network()
  with model:
      # ...

  sim = nengo_spinnaker.Simulator(model)
  sim.run(10.)

Nengo SpiNNaker Simulator
=========================

.. autoclass:: nengo_spinnaker.Simulator
   :members:
