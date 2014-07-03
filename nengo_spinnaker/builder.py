"""Build models into intermediate representations which can be simply
converted into PACMAN problem specifications.
"""

import math
import numpy as np

import nengo.utils.builder
from nengo.utils import distributions as dists
from nengo.utils.compat import is_integer
from nengo.utils.inspect import checked_call

import utils


class Builder(object):
    pre_rpn_transforms = list()  # Network transforms which alter connectivity
    post_rpn_transforms = list()  # Network transforms which alter objects

    @classmethod
    def register_connectivity_transform(cls, func):
        """Add a new network transform to the builder."""
        cls.pre_rpn_transforms.append(func)

    @classmethod
    def register_object_transform(cls, func):
        """Add a new network transform to the builder."""
        cls.post_rpn_transforms.append(func)

    @classmethod
    def build(cls, network, dt, seed):
        """Build an intermediate representation of a Nengo model which can be
        assembled to form a PACMAN problem graph.
        """
        # Flatten the network
        (objs, conns) = nengo.utils.builder.objs_and_connections(network)

        # Generate a RNG
        rng = np.random.RandomState(seed)

        # Apply all network transforms which modify connectivity, they should
        # occur before removing pass through nodes
        for transform in cls.pre_rpn_transforms:
            (objs, conns) = transform(objs, conns, network.probes)

        # Remove pass through nodes
        (objs, conns) = utils.builder.remove_passthrough_nodes(objs, conns)

        # Replace all connections with fully specified equivalents
        new_conns = list()
        for c in conns:
            new_conns.append(
                utils.builder.IntermediateConnection.from_connection(c))
        conns = new_conns

        # Apply all network transforms which modify/replace network objects
        for transform in cls.post_rpn_transforms:
            (objs, conns) = transform(objs, conns, network.probes, dt, rng)

        # Assign an ID to each object
        object_ids = dict([(o, i) for i, o in enumerate(objs)])

        # Create the keyspace for the model
        keyspace = _create_keyspace(conns)

        # Assign the keyspace to the connections, drill down as far as possible
        connection_ids = _get_outgoing_ids(conns)
        for c in conns:
            # Assign the keyspace if one isn't already set
            if c.keyspace is None:
                c.keyspace = keyspace()

            # Set fields within the keyspace
            c.keyspace = c.keyspace(o=object_ids[c.pre])
            if not c.keyspace.is_set_i:
                c.keyspace = c.keyspace(i=connection_ids[c])

        # Build the list of output keys for all of the ensemble objects now
        # that we've assigned IDs and keyspaces.
        for obj in objs:
            if isinstance(obj, IntermediateEnsemble):
                obj.create_output_keys(object_ids[obj], keyspace)

        # Return list of intermediate representation objects and connections
        return objs, conns, keyspace


def _create_keyspace(connections):
    """Create the minimum keyspace necessary to represent the connection set.
    """
    # Get connection IDs
    max_o = len(set([c.pre for c in connections]))
    max_i = max([i for i in _get_outgoing_ids(connections).values()])
    max_d = max([c.width for c in connections])

    # Get the number of bits necessary for these
    (bits_o, bits_i, bits_d) = [int(math.ceil(math.log(v + 1, 2)))
                                for v in [max_o, max_i, max_d]]

    # Ensure that these will fit within a 32-bit key
    padding = 32 - (1 + bits_o + bits_i + bits_d)
    assert padding >= 0
    bits_o += padding

    # Create the keyspace
    return utils.keyspaces.create_keyspace(
        'NengoDefault',
        [('x', 1), ('o', bits_o), ('i', bits_i), ('d', bits_d)],
        'xoi'
    )


def _get_outgoing_ids(connections):
    """Get the outgoing ID of each connection.
    """
    output_blocks = dict()
    connection_ids = dict()

    # Iterate through the connections building connection blocks where
    # necessary.
    for c in connections:
        if c.pre not in output_blocks:
            output_blocks[c.pre] =\
                (utils.connections.Connections() if not
                 isinstance(c.pre, IntermediateEnsemble) else
                 utils.connections.OutgoingEnsembleConnections())
        output_blocks[c.pre].add_connection(c)
        connection_ids[c] = output_blocks[c.pre][c]

    return connection_ids


class IntermediateEnsemble(object):
    def __init__(self, n_neurons, gains, bias, encoders, decoders,
                 eval_points, decoder_headers):
        self.n_neurons = n_neurons

        # Assert that the number of neurons is reflected in other parameters
        assert gains.size == n_neurons
        assert bias.size == n_neurons
        assert encoders.shape[0] == n_neurons

        # Get the number of dimensions represented and store fundamental
        # parameters
        self.size_in = self.n_dimensions = encoders.shape[1]
        self.gains = gains
        self.bias = bias
        self.encoders = encoders
        self.decoders = decoders

        # Output keys
        self.decoder_headers = decoder_headers
        self.output_keys = None

        # Recording parameters
        self.record_spikes = False
        self.record_voltage = False
        self.probes = list()

        # Direct input
        self.direct_input = np.zeros(self.n_dimensions)

    def create_output_keys(self, ens_id, keyspace):
        self.output_keys = list()
        for header in self.decoder_headers:
            ks = keyspace if header[0] is None else header[0]
            ks = ks(o=ens_id, i=header[1], d=header[2])
            self.output_keys.append(ks.key())


class IntermediateLIFEnsemble(IntermediateEnsemble):
    def __init__(self, n_neurons, gains, bias, encoders, decoders, tau_rc,
                 tau_ref, eval_points, decoder_headers):
        super(IntermediateLIFEnsemble, self).__init__(
            n_neurons, gains, bias, encoders, decoders, eval_points,
            decoder_headers)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    @classmethod
    def from_object(cls, ens, out_conns, dt, rng):
        assert isinstance(ens.neuron_type, nengo.neurons.LIF)
        assert isinstance(ens, nengo.Ensemble)

        if ens.seed is None:
            rng = np.random.RandomState(rng.tomaxint())
        else:
            rng = np.random.RandomState(ens.seed)

        # Generate evaluation points
        if ens.eval_points is None:
            dims, neurons = ens.dimensions, ens.n_neurons
            n_points = max(np.clip(500 * dims, 750, 2500), 2*neurons)
            eval_points = dists.UniformHypersphere(ens.dimensions).sample(
                n_points, rng=rng) * ens.radius
        elif is_integer(ens.eval_points):
            eval_points = dists.UniformHypersphere(ens.dimensions).sample(
                ens.eval_points, rng=rng) * ens.radius
        else:
            eval_points = np.array(ens.eval_points, dtype=np.float64)
            if eval_points.dim == 1:
                eval_points.shape = (-1, 1)

        # Generate gains, bias
        gain = ens.gain
        bias = ens.bias
        if gain is None or bias is None:
            if hasattr(ens.max_rates, 'sample'):
                ens.max_rates = ens.max_rates.sample(ens.n_neurons, rng)
            if hasattr(ens.intercepts, 'sample'):
                ens.intercepts = ens.intercepts.sample(ens.n_neurons, rng)
            (gain, bias) = ens.neuron_type.gain_bias(ens.max_rates,
                                                     ens.intercepts)

        # Generate encoders
        if ens.encoders is None:
            sphere = dists.UniformHypersphere(ens.dimensions, surface=True)
            encoders = sphere.sample(ens.n_neurons, rng=rng)
        else:
            encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.n_neurons, ens.size_in)

            if encoders.shape != enc_shape:
                # TODO Remove this when it is checked by Nengo
                raise nengo.builder.ShapeMismatch(
                    'Encoder shape is %s. Should be (n_neurons, dimensions); '
                    "in this case '%s'." % enc_shape, encoders.shape)

            norm = np.sum(encoders ** 2, axis=1)[:, np.newaxis]
            encoders /= np.sqrt(norm)

        # Generate decoders for outgoing connections
        decoders = list()
        tfses = utils.connections.OutgoingEnsembleConnections(out_conns)

        def build_decoder(function, evals, solver):
            """Internal function for building a single decoder."""
            if evals is None:
                evals = eval_points

            x = np.dot(evals, encoders.T / ens.radius)
            activities = ens.neuron_type.rates(x, gain, bias)

            if function is None:
                targets = evals
            else:
                (value, _) = checked_call(function, evals[0])
                function_size = np.asarray(value).size
                targets = np.zeros((len(evals), function_size))

                for i, ep in enumerate(evals):
                    targets[i] = ep

            if solver is None:
                solver = nengo.decoders.LstsqL2()

            return solver(activities, targets, rng)[0]

        decoder_builder = utils.decoders.DecoderBuilder(build_decoder)

        # Build each of the decoders in turn
        for tfse in tfses.transforms_functions:
            decoders.append(decoder_builder.get_transformed_decoder(
                tfse.function, tfse.transform, tfse.eval_points, tfse.solver))

        # Compress and merge the decoders
        (decoder_headers, decoders) =\
            utils.decoders.get_combined_compressed_decoders(decoders)
        decoders /= dt

        return cls(ens.n_neurons, gain, bias, encoders, decoders,
                   ens.neuron_type.tau_rc, ens.neuron_type.tau_ref,
                   eval_points, decoder_headers)


def build_ensembles(objects, connections, probes, dt, rng):
    """Build Ensembles and related connections into intermediate
    representation form.
    """
    new_objects = list()
    new_connections = list()

    # Sort out GlobalInhibitionConnections
    (objects, connections) = process_global_inhibition_connections(
        objects, connections, probes)

    # Create an intermediate representation for each Ensemble
    for obj in objects:
        if not isinstance(obj, nengo.Ensemble):
            new_objects.append(obj)
            continue

        # Build the appropriate intermediate representation for the Ensemble
        if isinstance(obj.neuron_type, nengo.neurons.LIF):
            # Get the set of outgoing connections for this Ensemble so that
            # decoders can be solved for.
            out_conns = [c for c in connections if c.pre == obj]
            new_obj = IntermediateLIFEnsemble.from_object(obj, out_conns,
                                                          dt, rng)
            new_objects.append(new_obj)
        else:
            raise NotImplementedError("nengo_spinnaker does not currently "
                                      "support '%s' neurons."
                                      % obj.neuron_type.__class__.__name__)

        # Modify connections into/out of this ensemble
        for c in connections:
            if c.pre == obj:
                c.pre = new_obj
            if c.post == obj:
                c.post = new_obj

        # Mark the Ensemble as recording spikes/voltages if appropriate
        for p in probes:
            if p.target == obj:
                if p.attr == 'spikes':
                    new_obj.record_spikes = True
                    new_obj.probes.append(p)
                elif p.attr == 'voltage':
                    raise NotImplementedError("Voltage probing not currently "
                                              "supported.")
                    new_obj.record_voltage = True
                    new_obj.probes.append(p)

    # Add direct inputs
    for c in connections:
        if (isinstance(c.post, IntermediateEnsemble) and
                isinstance(c.pre, nengo.Node) and not callable(c.pre.output)):
            # This Node just forms direct input, add it to direct input and
            # don't add the connection to the list of new connections
            inp = c.pre.output
            if c.function is not None:
                inp = c.function(inp)
            c.post.direct_input += np.dot(c.transform, inp)
        else:
            new_connections.append(c)

    return new_objects, new_connections

Builder.register_object_transform(build_ensembles)


class IntermediateGlobalInhibitionConnection(
        utils.builder.IntermediateConnection):
    @classmethod
    def from_connection(cls, c):
        # Assert that the transform is as we'd expect
        assert isinstance(c.post, nengo.objects.Neurons)
        assert np.all([c.transform[0] == t for t in c.transform])

        # Compress the transform to have output dimension of 1
        tr = c.transform[0][0]

        # Get the keyspace for the connection
        keyspace = getattr(c, 'keyspace', None)

        # Create a new instance
        return cls(c.pre, c.post.ensemble, c.synapse, c.function, tr, c.solver,
                   c.eval_points, keyspace)


def process_global_inhibition_connections(objs, connections, probes):
    # Go through connections replacing global inhibition connections with
    # an intermediate representation
    new_connections = list()
    for c in connections:
        if (isinstance(c.post, nengo.objects.Neurons) and
                np.all([c.transform[0] == t for t in c.transform])):
            # This is a global inhibition connection, swap out
            c = IntermediateGlobalInhibitionConnection.from_connection(c)
        new_connections.append(c)

    return objs, new_connections


class IntermediateProbe(object):
    def __init__(self, size_in, sample_every, probe):
        self.size_in = size_in
        self.sample_every = sample_every
        self.probe = probe


def insert_decoded_output_probes(objs, connections, probes):
    """Creates a new object representing decoded output probes and provides
    appropriate connections.
    """
    objs = objs
    connections = connections

    # Add new objects and connections for 'decoded output' probes
    for probe in probes:
        if probe.attr == 'decoded_output' or probe.attr == 'output':
            p = IntermediateProbe(probe.size_in, probe.sample_every, probe)

            # Create a new connection for this Node, if there is no transform
            # on the connection then we can create one on the assumption that
            # size_in and size_out are equivalent.
            conn_args = probe.conn_args
            if 'transform' not in conn_args:
                assert probe.target.size_out == p.size_in
                conn_args['transform'] = np.eye(p.size_in)
            c = utils.builder.IntermediateConnection(probe.target, p,
                                                     **probe.conn_args)

            # Add the new probe object and connection object
            objs.append(p)
            connections.append(c)

    return objs, connections

Builder.register_connectivity_transform(insert_decoded_output_probes)
