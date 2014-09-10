import collections
import sys
import numpy as np
import nengo
import nengo_spinnaker
import nengo_spinnaker.utils.profiling

spinnaker = True
dimensions = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 1
ensemble_size = int(sys.argv[2]) if __name__ == "__main__" and len(sys.argv) > 2 else 50
plot = False

from nengo.utils.functions import whitenoise
model = nengo.Network()
with model:
    config = nengo_spinnaker.Config()
        
    inp = nengo.Node(whitenoise(0.1, 5, dimensions=dimensions), label = "inp")
    config[inp].f_of_t = True
    
    pre = nengo.Ensemble(ensemble_size, dimensions=dimensions, label = "pre")
    nengo.Connection(inp, pre)
    post = nengo.Ensemble(ensemble_size, dimensions=dimensions, label = "post")
    conn = nengo.Connection(pre, post, function=lambda x: np.random.random(dimensions))
    config[post].profiler_num_samples = 10000
    assert config[post].profiler_num_samples == 10000
    
    #if not spinnaker:
    inp_p = nengo.Probe(inp)
        
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)
    
    import matplotlib.pyplot as plt
    if spinnaker:
        sim = nengo_spinnaker.Simulator(model, config = config)
        sim.run(0.1, clean = True)
       
        # **HACK** this is only correct for ensemble
        tag_names = collections.defaultdict(lambda: "Unknown", 
            { 0: "Timer", 1: "Input filter", 2: "Neuron update", 3: "Output" })
        
        print "Dimensions %u, Ensemble size %u" % (dimensions, ensemble_size)
        for vertex, subvertices in sim.profile_data[post].iteritems():
            for subvertex, data in subvertices.iteritems():
                plot_data = nengo_spinnaker.utils.profiling.plot_profile_data(data)
                
                if plot:
                    figure, axis = plt.subplots()
                
                for tag, times in plot_data.iteritems():
                    if plot:
                        for start_time, duration in zip(times[0], times[1]):
                            axis.bar(tag, duration, bottom = start_time, width = 1.0, linewidth = None)
                    
                    print "%s min:%fms, max:%fms, mean:%fms" % (tag_names[tag], np.amin(times[1]), np.amax(times[1]), np.mean(times[1]))
                
                if plot:
                    axis.set_xticks([0.5 + float(i) for i in plot_data])
                    axis.set_xticklabels([tag_names[i] for i in plot_data])
                
        
    else:
        sim = nengo.Simulator(model)
        sim.run(1.0)
    
    if plot:
        plt.figure(figsize=(12, 8))
        
        for d in range(dimensions):
            #if dimensions > 1:
            plt.subplot(dimensions, 1, d + 1)
            
            #if not spinnaker:
            plt.plot(sim.trange(), sim.data[inp_p].T[d], c='k', label='Input')
                
            plt.plot(sim.trange(), sim.data[pre_p].T[d], c='b', label='Pre')
            plt.plot(sim.trange(), sim.data[post_p].T[d], c='r', label='Post')
            plt.ylabel("Dimension 1")
            plt.legend(loc='best')
    
        plt.show()