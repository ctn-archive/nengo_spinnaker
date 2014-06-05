import nengo
import nengo_spinnaker
import numpy as np

model = nengo.Network()
with model:
    source = nengo.Node(np.sin)
    target = nengo.Ensemble(75, 2)

    c1 = nengo.Connection(source, target, transform=[[0.5], [-.25]])

    p = nengo.Probe(target, synapse=0.05)

sim = nengo_spinnaker.Simulator(model)
sim.run(10.)

# Plot the results
from matplotlib import pyplot as plt

ts = np.arange(0, 10., 0.001)
plt.plot(ts, sim.data[p].T)
plt.xlabel("Time / s")
plt.ylabel("Decoded Output")
plt.show(block=True)
