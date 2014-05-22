import nengo
import nengo_spinnaker
import numpy as np

model = nengo.Network()

def printout(t, v):
    print t, v

with model:
    a = nengo.Node(np.sin, size_in=0, size_out=1, label='input')
    e = nengo.Ensemble(nengo.LIF(100), 2)
    b = nengo.Node(printout, size_in=2, size_out=0, label='output')

    nengo.Connection(a, e, transform=[[-.5], [.25]])
    nengo.Connection(e, b, synapse=0.05)

# Configure `a` as being a function of time
config = nengo_spinnaker.Config()
config[a].f_of_t = True
config[a].f_period = 8*np.pi  # A lie!

# Create the simulation
sim = nengo_spinnaker.Simulator(model, config=config)
sim.run(20.)
