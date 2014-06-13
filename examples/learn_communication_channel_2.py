import numpy as np
import nengo
import nengo_spinnaker

spinnaker = True
dimensions = 1

from nengo.utils.functions import whitenoise
model = nengo.Network()
with model:
    config = nengo_spinnaker.Config()
        
    inp = nengo.Node(whitenoise(0.1, 5, dimensions=dimensions), label = "inp")
    config[inp].f_of_t = True
    
    pre = nengo.Ensemble(60, dimensions=dimensions, label = "pre")
    nengo.Connection(inp, pre)
    post = nengo.Ensemble(60, dimensions=dimensions, label = "post")
    conn = nengo.Connection(pre, post, function=lambda x: np.random.random(dimensions))
    
    error = nengo.Ensemble(60, dimensions=dimensions)
    error_p = nengo.Probe(error, synapse=0.03)
    # Error = pre - post
    nengo.Connection(pre, error)
    nengo.Connection(post, error, transform=-1)
    # Modulatory connections don't impart current
    error_conn = nengo.Connection(error, post, modulatory=True)
    # Add the learning rule to the connection
    conn.learning_rule = nengo.PES(error_conn, learning_rate=1.0)
    
    if not spinnaker:
        inp_p = nengo.Probe(inp)
        
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)
    
    if spinnaker:
        sim = nengo_spinnaker.Simulator(model, config = config)
        sim.run(10.0, clean=False)
    else:
        sim = nengo.Simulator(model)
        sim.run(10.0)
    
    

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    
    for d in range(dimensions):
        #if dimensions > 1:
        plt.subplot(dimensions, 1, d + 1)
        
        if not spinnaker:
            plt.plot(sim.trange(), sim.data[inp_p].T[d], c='k', label='Input')
            
        plt.plot(sim.trange(), sim.data[pre_p].T[d], c='b', label='Pre')
        plt.plot(sim.trange(), sim.data[post_p].T[d], c='r', label='Post')
        plt.ylabel("Dimension 1")
        plt.legend(loc='best')

    plt.show()