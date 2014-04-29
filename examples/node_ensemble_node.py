import nengo
import nengo_spinnaker
import numpy as np

model = nengo.Network("Node -> Ensemble -> Node")

with model:
    def in_f(t):
        val = np.sin(t)
        print("Input %.3f" % val)
        return val

    def out_f(t, val):
        print("Output %.3f" % val[0])

    a = nengo.Node(in_f, size_in=0, size_out=1, label="Input")
    b = nengo.Ensemble(nengo.LIF(100), 1)
    c = nengo.Node(out_f, size_in=1, size_out=0, label="Output")

    nengo.Connection(a, b, synapse=0.01)
    nengo.Connection(b, c, synapse=0.01, transform=-1.)

sim = nengo_spinnaker.Simulator(model)
sim.run(10.)
