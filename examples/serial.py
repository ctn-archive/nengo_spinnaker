import nengo
import nengo_spinnaker
import numpy as np

model = nengo.Network("Node -> Node -> Ensemble -> Node")

with model:
    def in_f(t):
        val = np.sin(t)
        return val

    def out_f(t, val):
        print("Output %.3f" % val[0])

    a = nengo.Node(in_f, size_in=0, size_out=1, label="Input")
    b = nengo.Node(lambda t, val : val * -1., size_in=1, size_out=1)
    c = nengo.Ensemble(100, 1)
    d = nengo.Node(out_f, size_in=1, size_out=0, label="Output")

    nengo.Connection(a, b)
    nengo.Connection(b, c, synapse=0.01)
    nengo.Connection(c, d, synapse=0.01)

io = nengo_spinnaker.io.Serial(protocol=nengo_spinnaker.io.SpIOUART)
sim = nengo_spinnaker.Simulator(model, io=io)
sim.run(10.)
