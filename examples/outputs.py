import nengo
import numpy as np

D = 2

model = nengo.Model()
with model:
    def printout(label, t, x):
        print label, t, ','.join(['%1.3f'%xx for xx in x])

    a = nengo.Ensemble(90, 1)
    a_out = nengo.Node(lambda t,x: printout('a', t, x), size_in=1)
    nengo.Connection(a, a_out, filter=0.1)

    b = nengo.Ensemble(90, 2)
    b_out = nengo.Node(lambda t,x: printout('b', t, x), size_in=2)
    nengo.Connection(b, b_out, filter=0.1)

    c = nengo.Ensemble(90, 5)
    c_out = nengo.Node(lambda t,x: printout('c', t, x), size_in=5)
    nengo.Connection(c, c_out, function=lambda x: [-1, -0.5, 0, 0.5, 1], filter=0.1)

import nengo_spinnaker
sim = nengo_spinnaker.Simulator(model)
sim.run(2)


