import nengo
import nengo_spinnaker
import numpy as np

model = nengo.Network()
with model:
    source = nengo.Node(np.sin)
    target = nengo.Ensemble(25, 1)

    c1 = nengo.Connection(source, target)

    p = nengo.Probe(target, 'spikes')

sim = nengo_spinnaker.Simulator(model)
sim.run(10.)

# Plot the results
from matplotlib import pyplot as plt

es = [[0.] for n in range(25)]
for (i, f) in enumerate(sim.data[p]):
    for n in f:
        es[n].append(i*0.001)

plt.eventplot(es, colors=[[0, 0, 1]])
plt.xlabel("Time / s")
plt.ylabel("Neurons")
plt.show(block=True)
