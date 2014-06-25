import nengo
import nengo_spinnaker
import numpy as np

model = nengo.Network()

def feedin(t):
    return np.array([np.sin(t), np.cos(t)])

with model:
    a = nengo.Node(feedin, label='input')
    e = nengo.Ensemble(100, 2)
    p = nengo.Probe(e)

    nengo.Connection(a, e)

# Configure `a` as being a function of time
config = nengo_spinnaker.Config()
config[a].f_of_t = True
config[a].f_period = 8*np.pi  # A lie!

# Create the simulation
sim = nengo_spinnaker.Simulator(model, config=config)
sim.run(20.)

from matplotlib import pyplot as plt
plt.plot(sim.trange(), sim.data[p])
plt.xlabel('Time / s')
plt.show()
