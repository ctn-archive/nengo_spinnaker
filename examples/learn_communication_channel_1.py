import numpy as np
import nengo
import nengo_spinnaker

spinnaker = True

from nengo.utils.functions import whitenoise
model = nengo.Network()
with model:
    inp = nengo.Node(whitenoise(0.1, 5, dimensions=2))
    pre = nengo.Ensemble(60, dimensions=2)
    nengo.Connection(inp, pre)
    post = nengo.Ensemble(60, dimensions=2)
    conn = nengo.Connection(pre, post, function=lambda x: np.random.random(2))
    
    if not spinnaker:
        inp_p = nengo.Probe(inp)
        
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)
    
    if spinnaker:
        sim = nengo_spinnaker.Simulator(model)
    else:
        sim = nengo.Simulator(model)
    
    sim.run(10.0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    
    if not spinnaker:
        plt.plot(sim.trange(), sim.data[inp_p].T[0], c='k', label='Input')
        
    plt.plot(sim.trange(), sim.data[pre_p].T[0], c='b', label='Pre')
    plt.plot(sim.trange(), sim.data[post_p].T[0], c='r', label='Post')
    plt.ylabel("Dimension 1")
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    
    if not spinnaker:
        plt.plot(sim.trange(), sim.data[inp_p].T[1], c='k', label='Input')
        
    plt.plot(sim.trange(), sim.data[pre_p].T[1], c='b', label='Pre')
    plt.plot(sim.trange(), sim.data[post_p].T[1], c='r', label='Post')
    plt.ylabel("Dimension 2")
    plt.legend(loc='best')
    plt.show()