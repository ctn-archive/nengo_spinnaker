"""
Builder for running Nengo on SpiNNaker

The following information is needed:

For each Ensemble:
    - id (for referring to it)
    - N  (number of neurons)
    - bias
    - tau_rc, tau_ref
    - encoders*gain
    - decoders (one large matrix for all decoders from the ensemble)
    - filters (list of low-pass filter tau values)
For each Connection:
    - pre
    - post
    - offset, length (in pre's decoder matrix)
    - target filter index (in post's filters list)
    
Note that connections from a Node into an Ensemble are handled by the
simulator, not the builder.   
"""    


import nengo
import numpy as np
import nengo.build_utils


class NodeData:
    """Constructed information for a Node."""
    def __init__(self, ens, rng, id):
        self.id = id
        self.filters = []
    def get_target(self, filter):
        if filter not in self.filters:
            self.filters.append(filter)
        return self.filters.index(filter)

class EnsembleData:
    """Constructed information for an Ensemble."""
    def __init__(self, ens, rng, id):
        self.id = id                # my identifier
        self.filters = []           # list of input tau filter values
        self.decoders_by_func = {}  # cache for decoders of different funcs
        self.decoder_list = []      # list of actual decoders (with transform)
        self.ens = ens              # the ensemble we're associated with
    
        self.N = ens.neurons.n_neurons     # number of neurons
        
        # Create random number generator
        if ens.seed is None:
            rng = np.random.RandomState(rng.tomaxint())
        else:
            rng = np.random.RandomState(ens.seed)
        self.rng = rng    
            
        # Generate eval points
        if ens.eval_points is None:
            # TODO: standardize how to set number of samples
            #  (this is different than the reference implementation!)
            S = min(ens.dimensions * 500, 5000)
            self.eval_points = nengo.decoders.sample_hypersphere(
                ens.dimensions, S, rng) * ens.radius
        else:
            self.eval_points = np.array(ens.eval_points, dtype=np.float64)
            if self.eval_points.ndim == 1:
                self.eval_points.shape = (-1, 1)
        
        # TODO: change this to not modify Model
        # Set up neurons
        if ens.neurons.gain is None or ens.neurons.bias is None:
            # if max_rates and intercepts are distributions,
            # turn them into fixed samples.
            if hasattr(ens.max_rates, 'sample'):
                ens.max_rates = ens.max_rates.sample(
                    ens.neurons.n_neurons, rng=rng)
            if hasattr(ens.intercepts, 'sample'):
                ens.intercepts = ens.intercepts.sample(
                    ens.neurons.n_neurons, rng=rng)
            ens.neurons.set_gain_bias(ens.max_rates, ens.intercepts)
        self.bias = ens.neurons.bias
        self.gain = ens.neurons.gain
        self.tau_rc = ens.neurons.tau_rc
        self.tau_ref = ens.neurons.tau_ref
        
            
        # Set up encoders
        if ens.encoders is None:
            self.encoders = ens.neurons.default_encoders(ens.dimensions, rng)
        else:
            self.encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
            if self.encoders.shape != enc_shape:
                raise ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions);"
                    " in this case %s." % (self.encoders.shape, enc_shape))

            norm = np.sum(self.encoders ** 2, axis=1)[:, np.newaxis]
            self.encoders /= np.sqrt(norm)
        ens.encoders = self.encoders   #TODO: remove this when it is no longer
                                       # required be Ensemble.activities()        
            
    def get_target(self, filter):
        """Return the index of the input that has this filter value
        
        If that filter value does not yet exist, it is added.
        """
        if filter not in self.filters:
            self.filters.append(filter)
        return self.filters.index(filter)
        
    def get_decoder_location(self, c):
        """Return the starting index of the decoder for this connection
        
        If it already exists, the existing index is returned.  Otherwise it
        is added to the decoder_list.
        """
        
        # check if the decoder already exists for this function and transform
        start = 0
        for d,t,f in self.decoder_list:
            if t == c.transform and c.function == f:
                return start
            start += d.shape[1]    
                
        # check if the decoder already exists for this function       
        decoder = self.decoders_by_func.get(c.function, None)
        if decoder is None:
            # compute the decoder
            eval_points = c.eval_points
            if eval_points is None:
                eval_points = self.eval_points
            activities = self.ens.activities(eval_points)
            if c.function is None:
                targets = eval_points
            else:
                targets = np.array(
                            [c.function(ep) for ep in eval_points])
                if targets.ndim < 2:
                    targets.shape = targets.shape[0], 1
            decoder = c.decoder_solver(activities, targets, self.rng)
            self.decoders_by_func[c.function] = decoder
        
        # combine the decoder with the transform and record it in the list
        decoder = np.dot(decoder, c.transform.T)
        self.decoder_list.append((decoder, c.transform, c.function))
        return start
        

            
    def text_data(self):
        r = []
        r.append('  id=%d'%self.id)
        r.append('  N=%d'%self.N)
        r.append('  tau_rc=%g'%self.tau_rc)
        r.append('  tau_ref=%g'%self.tau_ref)
        r.append('  bias=%s'%self.bias)
        r.append('  encoders=%s'%(self.encoders*self.gain[:,None]))
        r.append('  decoders=%s'%(self.decoder_list))
        
        return '\n'.join(r)
        
class OnboardConnectionData:
    def __init__(self, c, pre, post):
        self.pre = pre.id
        self.post = post.id
        self.offset = pre.get_decoder_location(c)
        self.length = c.dimensions
        self.target_buffer = post.get_target(c.filter)
        
class ExternalConnectionData:
    def __init__(self, c, pre, post):
        self.pre = pre.id
        self.post = post.id
        self.target_buffer = post.get_target(c.filter)
        


class Builder:
    def __init__(self, model, dt=0.001, seed=None):    
        rng = np.random.RandomState(seed)
    
        # Get rid of all passthrough nodes, replacing them with the equivalent
        #   connections
        objs, connections = nengo.build_utils.remove_passthrough_nodes(
                                                model.objs, model.connections)
        
        self.ensembles = {}  # just ensembles
        self.nodes = {}      # just nodes
        self.items = {}      # both ensembles and nodes
        
        # parse the objects
        for obj in objs:
            if isinstance(obj, nengo.Ensemble):
                self.ensembles[obj] = EnsembleData(obj, rng, 
                                                        id=len(self.ensembles))
                self.items[obj] = self.ensembles[obj]
            elif isinstance(obj, nengo.Node):
                self.nodes[obj] = NodeData(obj, rng, id=-1-len(self.nodes))
                self.items[obj] = self.nodes[obj]
            else:
                raise Exception('Unknown object in model: %s'%obj)
        
        
        self.onboard_connections = []   # connections handled by SpiNNaker
        self.external_connections = []  # connections handled by host
        
        # parse the connections
        for c in connections:
            if isinstance(c.pre, nengo.Ensemble):
                data = OnboardConnectionData(c, self.ensembles[c.pre], 
                                                        self.items[c.post])
                self.onboard_connections.append(data)
                
            elif isinstance(c.pre, nengo.Node):
                self.external_connections.append(ExternalConnectionData(c, 
                        self.nodes[c.pre], self.items[c.post]))
        

    def print_ensembles(self):        
        for ens in self.ensembles.values():
            print ens.text_data()
            
    def print_connections(self):        
        for c in self.onboard_connections:
            print c.pre, c.post, c.offset, c.length, c.target_buffer
