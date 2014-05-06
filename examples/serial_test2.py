import nengo
import numpy as np

D = 2

model = nengo.Network('Communication Channel')
with model:
    input = nengo.Node(lambda t: [np.sin(t), np.cos(t)], label='input')
    a = nengo.Ensemble(nengo.LIF(100), D, label='a')
    b = nengo.Ensemble(nengo.LIF(100), D, label='b')
    def printout(t, x):
        print 1, t, x
        return []
    output = nengo.Node(printout, size_in=D, label='output')

    osc = nengo.Ensemble(nengo.LIF(100), 2, label='osc')
    nengo.Connection(osc, osc, transform=[[1.1, 0.1], [-0.1, 1.1]],
            synapse=0.1)


    def printout2(t, x):
        #print 2, t, x
        return []

    output2 = nengo.Node(printout2, size_in=D, label='output')
    nengo.Connection(osc, output2, synapse=0.03)

    nengo.Connection(input, a, synapse=0.01)
    nengo.Connection(a, b, synapse=0.01)
    #nengo.Connection(b, b, transform=0.9, synapse=0.1)
    nengo.Connection(b, output, synapse=0.01)

import nengo_spinnaker
sim = nengo_spinnaker.Simulator(model, seed=3, use_serial=True)

# sim.builder.print_connections()
#sim = nengo.Simulator(model)
sim.run(100)


