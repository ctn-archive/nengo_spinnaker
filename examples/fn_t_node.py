import nengo
import nengo_spinnaker
import numpy as np

model = nengo.Network()

def printout(t, v):
    print t, v

def feedin(t):
    return np.array([np.sin(t), np.cos(t)])

with model:
    a = nengo.Node(feedin, label='input')
    e = nengo.Ensemble(100, 2)
    b = nengo.Node(printout, size_in=2, size_out=0, label='output')

    nengo.Connection(a, e)
    nengo.Connection(e, b, synapse=0.05, transform=[[0.5, -0.5]])

# Configure `a` as being a function of time
config = nengo_spinnaker.Config()
config[a].f_of_t = True
config[a].f_period = 8*np.pi  # A lie!

# Create the simulation
sim = nengo_spinnaker.Simulator(model, config=config)
sim.run(20.)
