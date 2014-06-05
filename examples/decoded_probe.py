import nengo
import numpy as np

# Construct the model
model = nengo.Network()

with model:
    i = nengo.Node(np.sin)
    a = nengo.Ensemble(nengo.LIF(100), 2)
    p = nengo.Probe(a)

    nengo.Connection(i, a, transform=[[0.5], [-0.25]])

# Simulate
import nengo_spinnaker
sim = nengo_spinnaker.Simulator(model)
sim.run(10.)

# Plot the output
from matplotlib import pyplot as plt

ts = np.arange(0, 10., step=0.001)
plt.plot(ts, sim.data[p].T)
plt.show()
