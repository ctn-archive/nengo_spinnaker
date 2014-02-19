import nengo
import numpy as np

D = 10

model = nengo.Model('Basal Ganglia')
with model:
    input = nengo.Node([1]*D, label='input')
    def printout(t, x):
        print t, x
    output = nengo.Node(printout, size_in=D, label='output')
    bg = nengo.networks.BasalGanglia(D, 20, label='BG')
    nengo.Connection(input, bg.input, filter=0.01)
    nengo.Connection(bg.output, output, filter=0.01)
    
import nengo_spinnaker
sim = nengo_spinnaker.Simulator(model)

sim.builder.print_connections()
#sim = nengo.Simulator(model)
#sim.run(0.1)


