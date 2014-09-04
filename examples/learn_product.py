import numpy as np
import nengo
import nengo_spinnaker
import matplotlib.pyplot as plt
from nengo.utils.functions import whitenoise

model = nengo.Network()
config = nengo_spinnaker.Config()

with model:
    # -- input and pre popluation
    inp = nengo.Node(whitenoise(0.1, 5, dimensions=2))
    config[inp].f_of_t = True
    
    pre = nengo.Ensemble(120, dimensions=2)
    nengo.Connection(inp, pre)
    
    # -- error population
    prod_node = nengo.Node(lambda t, x: x[0] * x[1], size_in=2)  # We'll give it the actual product
    config[prod_node].f_of_t = True
    
    nengo.Connection(inp, prod_node, synapse=None)
    error = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(prod_node, error)
    
    # -- inhibit error after 40 seconds
    inhib = nengo.Node(lambda t: 2.0 if t > 40.0 else 0.0)
    config[inhib].f_of_t = True
    
    nengo.Connection(inhib, error.neurons, transform=[[-1]] * error.n_neurons)

    # -- post population
    post = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(post, error, transform=-1)
    error_conn = nengo.Connection(error, post, modulatory=True)
    nengo.Connection(pre, post,
                     function=lambda x: np.random.random(1),
                     learning_rule=nengo.PES(error_conn, learning_rate=1.0))
    
    # -- probes
    prod_p = nengo.Probe(prod_node)
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)
    error_p = nengo.Probe(error, synapse=0.03)

    sim = nengo_spinnaker.Simulator(model, config = config)
    sim.run(60)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), sim.data[pre_p], c='b')
plt.legend(('Pre decoding',), loc='best')
plt.subplot(3, 1, 2)
plt.plot(sim.trange(), sim.data[prod_p], c='k', label='Actual product')
plt.plot(sim.trange(), sim.data[post_p], c='r', label='Post decoding')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.plot(sim.trange(), sim.data[error_p], c='b')
plt.ylim(-1, 1)
plt.legend(("Error",), loc='best')

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sim.trange()[:2000], sim.data[pre_p][:2000], c='b')
plt.legend(('Pre decoding',), loc='best')
plt.subplot(3, 1, 2)
plt.plot(sim.trange()[:2000], sim.data[prod_p][:2000], c='k', label='Actual product')
plt.plot(sim.trange()[:2000], sim.data[post_p][:2000], c='r', label='Post decoding')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.plot(sim.trange()[:2000], sim.data[error_p][:2000], c='b')
plt.ylim(-1, 1)
plt.legend(("Error",), loc='best')

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sim.trange()[38000:42000], sim.data[pre_p][38000:42000], c='b')
plt.legend(('Pre decoding',), loc='best')
plt.subplot(3, 1, 2)
plt.plot(sim.trange()[38000:42000], sim.data[prod_p][38000:42000], c='k', label='Actual product')
plt.plot(sim.trange()[38000:42000], sim.data[post_p][38000:42000], c='r', label='Post decoding')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.plot(sim.trange()[38000:42000], sim.data[error_p][38000:42000], c='b')
plt.ylim(-1, 1)
plt.legend(("Error",), loc='best')