import nengo
from nengo.utils.matplotlib import rasterplot
import numpy as np

model = nengo.Network()
with model:
    a = nengo.Node(.6)
    b = nengo.Node(lambda t: 0. if t < 5. else 1.)


    intercepts = nengo.objects.Uniform(0.3, 1)
    d = nengo.Ensemble(50, 1, intercepts=intercepts, label="Gate")
    e = nengo.Ensemble(100, 1, label="Gated")
    f = nengo.Ensemble(200, 1, label="Output")

    pf = nengo.Probe(f, synapse=0.05)
    ps = nengo.Probe(f, 'spikes')

    nengo.Connection(a, e)
    nengo.Connection(e, f)
    nengo.Connection(b, d)
    nengo.Connection(d, e.neurons, transform=[[-10.]]*100)

import nengo_spinnaker
config = nengo_spinnaker.Config()
config[b].f_of_t = True
sim = nengo_spinnaker.Simulator(model, config=config)
sim.run(10., clean=False)

sim_ = nengo.Simulator(model)
sim_.run(10.)

from matplotlib import pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(sim_.trange(), sim_.data[pf], label='reference')
plt.plot(sim.trange(), sim.data[pf], label='SpiNNaker')
plt.legend()
plt.ylim((-.1, 0.75))
plt.xlabel('Time / s')
plt.ylabel('Decoded Output')

plt.savefig('inhibition.png')
plt.show(block=True)
