import nengo
import numpy as np

model = nengo.Network()
with model:
    a = nengo.Node(0.5)
    b = nengo.Node(lambda t: 0. if t < 5. else 1.)


    intercepts = nengo.objects.Uniform(0.3, 1)
    d = nengo.Ensemble(50, 1, intercepts=intercepts)
    e = nengo.Ensemble(50, 1)
    f = nengo.Ensemble(50, 1)

    pf = nengo.Probe(d, synapse=0.05)
    ps = nengo.Probe(f, 'spikes')

    nengo.Connection(a, e)
    nengo.Connection(e, f)
    nengo.Connection(b, d)
    nengo.Connection(d, e.neurons, transform=[[-100.]]*50)

import nengo_spinnaker
config = nengo_spinnaker.Config()
config[b].f_of_t = True
sim = nengo_spinnaker.Simulator(model, config=config)
sim.run(10., clean=False)

sim_ = nengo.Simulator(model)
sim_.run(10.)

from matplotlib import pyplot as plt
import mpltools.style
mpltools.style.use('ggplot')

plt.figure()
plt.subplot(211)
plt.plot(sim_.trange(), sim_.data[pf], label='reference')
plt.plot(sim.trange(), sim.data[pf], label='SpiNNaker')
plt.legend()
plt.ylim((-.5, 1.0))
plt.xlabel('Time / s')
plt.ylabel('Decoded Output')

plt.subplot(212)
plt.eventplot(sim.data[ps], colors=[[1, 0, 0]])
plt.ylim((-.5, 49.5))
plt.xlabel('Time / s')
plt.ylabel('Neuron')

plt.axvline(5.0)
plt.show(block=True)
