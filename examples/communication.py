import nengo
import numpy as np

D = 2

model = nengo.Model('Communication Channel')
with model:
    input = nengo.Node(0.5, label='input')
    #a = nengo.Ensemble(100, D, label='a')
    b = nengo.Ensemble(9, D, label='b')
    def printout(t, x):
        print t, x
    output = nengo.Node(printout, size_in=D, label='output')
    
    nengo.Connection(input, b, filter=0.01, transform=[[1]]*D)
    #nengo.Connection(a, b, filter=0.01)
    #nengo.Connection(b, b, transform=0.9, filter=0.1)
    nengo.Connection(b, output, filter=0.01)
    
import nengo_spinnaker
sim = nengo_spinnaker.Simulator(model)

# sim.builder.print_connections()
#sim = nengo.Simulator(model)
sim.run(0.1)


