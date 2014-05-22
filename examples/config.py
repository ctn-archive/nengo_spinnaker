import nengo
import numpy as np

model = nengo.Model()
with model:
    input = nengo.Node(np.sin, label='input')
    a = nengo.Ensemble(100, 1, label='a')
    def printout(t, x):
        print t, x
    output = nengo.Node(printout, size_in=1, label='output')
    
    nengo.Connection(input, a, filter=0.01)
    nengo.Connection(a, output, filter=0.01)

import nengo_spinnaker
config = nengo_spinnaker.Config()
config[input].aplx = 'filename'
    
sim = nengo_spinnaker.Simulator(model, config)

#sim.run(0.1)


