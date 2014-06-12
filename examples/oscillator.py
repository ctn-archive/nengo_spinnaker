import nengo
from nengo.utils.functions import piecewise
import nengo_spinnaker
import numpy as np

model = nengo.Network()
with model:
    neurons = nengo.Ensemble(100, 2)
    i_f = nengo.Node(piecewise({0: [1, 0], 0.1: [0, 0]}))
    nengo.Connection(i_f, neurons)

    nengo.Connection(neurons, neurons,
                     transform=np.array([[1, 1], [-1, 1]])*1.1, synapse=0.05)

    p = nengo.Probe(neurons, synapse=0.05)

config = nengo_spinnaker.Config()
config[i_f].f_of_t = True  # This is pretty important!

sim = nengo_spinnaker.Simulator(model, config=config)
sim.run(10.)

from matplotlib import pyplot as plt
plt.figure()

plt.subplot(211)
plt.plot(sim.trange(), sim.data[p], label="SpiNNaker")
plt.legend()
plt.xlabel("Time / s")
plt.ylim([-1, 1])

sim = nengo.Simulator(model)
sim.run(10.)

plt.subplot(212)
plt.plot(sim.trange(), sim.data[p], label="Nengo Reference")
plt.legend()
plt.xlabel("Time / s")
plt.ylim([-1, 1])

plt.show(block=True)
