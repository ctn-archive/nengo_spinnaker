import numpy as np
import nengo
import nengo_spinnaker

spinnaker = True
dimensions = 1
learning_rates = [1.0, 0.5, 0.25]
post_colours = ['r', 'g', 'y']

from nengo.utils.functions import whitenoise
model = nengo.Network()
with model:
    config = nengo_spinnaker.Config()
        
    inp = nengo.Node(whitenoise(0.1, 5, dimensions=dimensions), label = "inp")
    config[inp].f_of_t = True
    
    pre = nengo.Ensemble(60, dimensions=dimensions, label = "pre")
    nengo.Connection(inp, pre)
    
    posts = [nengo.Ensemble(60, dimensions=dimensions, label = "post%u" % i) for i, l in enumerate(learning_rates)]
    conns = [nengo.Connection(pre, p, function=lambda x: np.random.random(dimensions)) for p in posts]
    errors = [nengo.Ensemble(60, dimensions=dimensions, label = "error%u" % i) for i, l in enumerate(learning_rates)]
    error_probes = [nengo.Probe(e, synapse=0.03) for e in errors]
    
    
    for e, p, c, l in zip(errors, posts, conns, learning_rates):
        # Error = pre - post
        nengo.Connection(pre, e)
        nengo.Connection(p, e, transform=-1)
        
        # Modulatory connections don't impart current
        error_conn = nengo.Connection(e, p, modulatory=True)
        
        # Add the learning rule to the connection
        c.learning_rule = nengo.PES(error_conn, learning_rate = l)

    inp_p = nengo.Probe(inp)
        
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_probes = [nengo.Probe(p, synapse=0.01) for p in posts]
    
    if spinnaker:
        sim = nengo_spinnaker.Simulator(model, config = config)
        sim.run(10.0, clean = True)
    else:
        sim = nengo.Simulator(model)
        sim.run(10.0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    
    for d in range(dimensions):
        plt.subplot(dimensions, 1, d + 1)

        plt.plot(sim.trange(), sim.data[inp_p].T[d], c='k', label='Input')
            
        plt.plot(sim.trange(), sim.data[pre_p].T[d], c='b', label='Pre')
        
        for p, l, c in zip(post_probes, learning_rates, post_colours):
            plt.plot(sim.trange(), sim.data[p].T[d], c=c, label='Post - Learning rate %f' % l)
        
        plt.ylabel("Dimension 1")
        plt.legend(loc='best')

    plt.show()