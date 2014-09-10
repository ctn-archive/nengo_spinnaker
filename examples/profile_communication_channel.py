import collections
import csv
import sys
import numpy as np
import nengo
import nengo_spinnaker
import nengo_spinnaker.utils.profiling
from nengo.utils.functions import whitenoise

dimensions = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 1
ensemble_size = int(sys.argv[2]) if __name__ == "__main__" and len(sys.argv) > 2 else 50

num_learning_rules = int(sys.argv[3]) if __name__ == "__main__" and len(sys.argv) > 3 else 0
num_post_populations = max(1, num_learning_rules)
learning_rates = [1.0 / float(i + 1) for i in range(num_learning_rules)]

profile_file = sys.argv[4] if __name__ == "__main__" and len(sys.argv) > 4 else "profile.csv"

# **HACK** this is only correct for ensemble
num_tags = 4

print "Dimensions %u, Ensemble size %u, Learning rules %u (post populations %u)" % (dimensions, ensemble_size, num_learning_rules, num_post_populations)

model = nengo.Network()
with model:
    config = nengo_spinnaker.Config()
        
    inp = nengo.Node(whitenoise(0.1, 5, dimensions=dimensions), label = "inp")
    inp_p = nengo.Probe(inp)
    config[inp].f_of_t = True
    
    pre = nengo.Ensemble(ensemble_size, dimensions=dimensions, label = "pre")
    pre_p = nengo.Probe(pre, synapse=0.01)
    nengo.Connection(inp, pre)
    
    posts = [nengo.Ensemble(ensemble_size, dimensions=dimensions, label = "post%u" % i) for i in range(num_post_populations)]
    posts_p = [nengo.Probe(p, synapse = 0.01) for p in posts]
    conns = [nengo.Connection(pre, p, function=lambda x: np.random.random(dimensions)) for p in posts]
    
    # Profile pre ensemble
    config[pre].profiler_num_samples = 10000
    
    if len(learning_rates) > 0:
        errors = [nengo.Ensemble(ensemble_size, dimensions=dimensions, label = "error%u" % i) for i, l in enumerate(learning_rates)]
        for e, p, c, l in zip(errors, posts, conns, learning_rates):
            # Error = pre - post
            nengo.Connection(pre, e)
            nengo.Connection(p, e, transform=-1)
            
            # Modulatory connections don't impart current
            error_conn = nengo.Connection(e, p, modulatory=True)
            
            # Add the learning rule to the connection
            c.learning_rule = nengo.PES(error_conn, learning_rate = l)

    sim = nengo_spinnaker.Simulator(model, config = config)
    sim.run(0.1, clean = True)
    
    with open(profile_file, "ab") as csvFile:
        csv_writer = csv.writer(csvFile)
        for vertex, subvertices in sim.profile_data[pre].iteritems():
            for subvertex, data in subvertices.iteritems():
                # Get subvertex's placement
                (x, y, p) = subvertex.placement.processor.get_coordinates()
                
                # Extract profiling data from this core
                plot_data = nengo_spinnaker.utils.profiling.plot_profile_data(data)
                
                csv_row = [dimensions, ensemble_size, num_learning_rules, x, y, p]
                csv_row.extend([None for i in range(num_tags)])
                
                for tag, times in plot_data.iteritems():
                    csv_row[tag + 6] = np.mean(times[1])
                
                csv_writer.writerow(csv_row)
