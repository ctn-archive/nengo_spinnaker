import numpy as np
import warnings

import nengo
from nengo.utils import distributions as dists
import nengo.utils.builder
import nengo.utils.numpy as npext
from nengo.utils.inspect import checked_call

import connection
import utils


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
            out_conns = [c for c in connections if c.pre_obj == obj]
            new_obj = IntermediateEnsembleLIF.from_object(obj, out_conns,
                                                          dt, rng)
            new_objects.append(new_obj)
        else:
            raise NotImplementedError("nengo_spinnaker does not currently "
                                      "support '%s' neurons."
                                      % obj.neuron_type.__class__.__name__)

        # Modify connections into/out of this ensemble
        for c in connections:
            if c.pre_obj is obj:
                c.pre_obj = new_obj
            if c.post_obj is obj:
                c.post_obj = new_obj

        # Mark the Ensemble as recording spikes/voltages if appropriate
        for p in probes:
            if p.target is obj:
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
        if (isinstance(c.post_obj, IntermediateEnsemble) and
                isinstance(c.pre_obj, nengo.Node) and not
                callable(c.pre_obj.output)):
            # This Node just forms direct input, add it to direct input and
            # don't add the connection to the list of new connections
            inp = c.pre_obj.output
            if c.function is not None:
                inp = c.function(inp)
            c.post_obj.direct_input += np.dot(c.transform, inp)
        else:
            new_connections.append(c)

    return new_objects, new_connections


def process_global_inhibition_connections(objs, connections, probes):
    # Go through connections replacing global inhibition connections with
    # an intermediate representation
    new_connections = list()
    for c in connections:
        if (isinstance(c.post_obj, nengo.ensemble.Neurons) and
                np.all([c.transform[0] == t for t in c.transform])):
            # This is a global inhibition connection, swap out
            c = IntermediateGlobalInhibitionConnection.from_connection(c)
        new_connections.append(c)

    return objs, new_connections


class IntermediateEnsemble(object):
    def __init__(self, n_neurons, gains, bias, encoders, decoders,
                 eval_points, decoder_headers, learning_rules, label=None):
        self.n_neurons = n_neurons
        self.label = label

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

        # Learning rules
        self.learning_rules = learning_rules

        # Recording parameters
        self.record_spikes = False
        self.record_voltage = False
        self.probes = list()

        # Direct input
        self.direct_input = np.zeros(self.n_dimensions)

    def create_output_keyspaces(self, ens_id, keyspace):
        self.output_keyspaces = list()
        for header in self.decoder_headers:
            ks = keyspace if header[0] is None else header[0]
            ks = ks(o=ens_id, i=header[1], d=header[2])
            self.output_keyspaces.append(ks)


class IntermediateEnsembleLIF(IntermediateEnsemble):
    def __init__(self, n_neurons, gains, bias, encoders, decoders, tau_rc,
                 tau_ref, eval_points, decoder_headers, learning_rules):
        super(IntermediateEnsembleLIF, self).__init__(
            n_neurons, gains, bias, encoders, decoders, eval_points,
            decoder_headers, learning_rules)
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
        if isinstance(ens.eval_points, dists.Distribution):
            n_points = ens.n_eval_points
            if n_points is None:
                n_points = nengo.utils.builder.default_n_eval_points(
                    ens.n_neurons, ens.dimensions)
            eval_points = ens.eval_points.sample(n_points, ens.dimensions, rng)
            eval_points *= ens.radius
        else:
            if (ens.eval_points is not None and
                    ens.eval_points.shape[0] != ens.n_eval_points):
                warnings.warn("Number of eval points doesn't match "
                              "n_eval_points.  Ignoring n_eval_points.")
            eval_points = np.array(ens.eval_points, dtype=np.float64)

        # Determine max_rates and intercepts
        if isinstance(ens.max_rates, dists.Distribution):
            max_rates = ens.max_rates.sample(ens.n_neurons, rng=rng)
        else:
            max_rates = np.array(max_rates)
        if isinstance(ens.intercepts, dists.Distribution):
            intercepts = ens.intercepts.sample(ens.n_neurons, rng=rng)
        else:
            intercepts = np.array(intercepts)

        # Generate gains, bias
        gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)

        # Generate encoders
        if isinstance(ens.encoders, dists.Distribution):
            encoders = ens.encoders.sample(ens.n_neurons, ens.dimensions,
                                           rng=rng)
        else:
            encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
            encoders /= npext.norm(encoders, axis=1, keepdims=True)

        # Generate decoders for outgoing connections
        decoders = list()
        tfses = utils.connections.OutgoingEnsembleConnections(out_conns)

        def build_decoder(function, evals, solver):
            """Internal function for building a single decoder."""
            if evals is None:
                evals = npext.array(eval_points, min_dims=2)
            else:
                evals = npext.array(evals, min_dims=2)

            assert solver is None or not solver.weights

            x = np.dot(evals, encoders.T / ens.radius)
            activities = ens.neuron_type.rates(x, gain, bias)

            if function is None:
                targets = evals
            else:
                (value, _) = checked_call(function, evals[0])
                function_size = np.asarray(value).size
                targets = np.zeros((len(evals), function_size))

                for i, ep in enumerate(evals):
                    targets[i] = function(ep)

            if solver is None:
                solver = nengo.solvers.LstsqL2()

            return solver(activities, targets, rng=rng)[0]

        decoder_builder = utils.decoders.DecoderBuilder(build_decoder)

        # Build each of the decoders in turn
        for tfse in tfses.transforms_functions:
            decoders.append(decoder_builder.get_transformed_decoder(
                tfse.function, tfse.transform, tfse.eval_points, tfse.solver))

        # Build list of learning rule, connection-index tuples
        learning_rules = list()
        for c in tfses:
            for l in utils.connections.get_learning_rules(c):
                learning_rules.append((l, tfses[c]))

        # By default compress all decoders
        decoders_to_compress = [True for d in decoders]

        # Turn off compression for all decoders associated with learning rules
        for l in learning_rules:
            decoders_to_compress[l[1]] = False

        # Compress and merge the decoders
        (decoder_headers, decoders) =\
            utils.decoders.get_combined_compressed_decoders(
                decoders, compress=decoders_to_compress)
        decoders /= dt

        return cls(ens.n_neurons, gain, bias, encoders, decoders,
                   ens.neuron_type.tau_rc, ens.neuron_type.tau_ref,
                   eval_points, decoder_headers, learning_rules)


class IntermediateGlobalInhibitionConnection(
        connection.IntermediateConnection):
    @classmethod
    def from_connection(cls, c):
        # Assert that the transform is as we'd expect
        assert isinstance(c.post_obj, nengo.ensemble.Neurons)
        assert np.all([c.transform[0] == t for t in c.transform])

        # Compress the transform to have output dimension of 1
        tr = c.transform[0][0]

        # Get the keyspace for the connection
        keyspace = getattr(c, 'keyspace', None)

        # Create a new instance
        return cls(c.pre_obj, c.post_obj.ensemble, c.synapse, c.function, tr,
                   c.solver, c.eval_points, keyspace)


class EnsembleLIF(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_ensemble'
    MAX_ATOMS = 128
    spikes_recording_region = 15

    def __init__(self, n_neurons, system_region, bias_region, encoders_region,
                 decoders_region, output_keys_region, input_filter_region,
                 input_filter_routing, inhib_filter_region,
                 inhib_filter_routing, gain_region, modulatory_filter_region,
                 modulatory_filter_routing, pes_region, spikes_region):
        super(EnsembleLIF, self).__init__(n_neurons)

        # Create regions
        self.regions = [None]*16
        self.regions[0] = system_region
        self.regions[1] = bias_region
        self.regions[2] = encoders_region
        self.regions[3] = decoders_region
        self.regions[4] = output_keys_region
        self.regions[5] = input_filter_region
        self.regions[6] = input_filter_routing
        self.regions[7] = inhib_filter_region
        self.regions[8] = inhib_filter_routing
        self.regions[9] = gain_region
        self.regions[10] = modulatory_filter_region
        self.regions[11] = modulatory_filter_routing
        self.regions[12] = pes_region
        self.regions[14] = spikes_region
        self.probes = list()

    @classmethod
    def assemble(cls, ens, assembler):
        # Prepare the system region
        system_items = [
            ens.n_dimensions,
            len(ens.output_keyspaces),
            ens.n_neurons,
            assembler.timestep,
            int(ens.tau_ref / (assembler.timestep * 10**-6)),
            utils.fp.bitsk(assembler.dt / ens.tau_rc),
            0x1 if ens.record_spikes else 0x0,
            1
        ]

        # Prepare the input filtering regions
        # Prepare the inhibitory filtering regions
        in_conns = assembler.get_incoming_connections(ens)
        inhib_conns =\
            [c for c in in_conns if
             isinstance(c, IntermediateGlobalInhibitionConnection)]
        modul_conns = [c for c in in_conns if c.modulatory]
        input_conns = [c for c in in_conns
                       if c not in inhib_conns and c not in modul_conns]

        (input_filter_region, input_filter_routing, _) =\
            utils.vertices.make_filter_regions(input_conns, assembler.dt)
        (inhib_filter_region, inhib_filter_routing, _) =\
            utils.vertices.make_filter_regions(inhib_conns, assembler.dt)
        (modul_filter_region, modul_filter_routing, modul_filter_assign) =\
            utils.vertices.make_filter_regions(modul_conns, assembler.dt)

        # From list of learning rules, extract list of PES learning rules
        pes_learning_rules = [l for l in ens.learning_rules
                              if isinstance(l[0], nengo.PES)]

        # Check no non-supported learning rules were present in original list
        if len(pes_learning_rules) != len(ens.learning_rules):
            raise NotImplementedError("Only PES learning rules currently "
                                      "supported.")

        # Begin PES items with number of learning rules
        pes_items = [len(pes_learning_rules)]
        for p in pes_learning_rules:
            # Generate block of data for this learning rule
            data = [
                utils.fp.bitsk(p[0].learning_rate * assembler.dt),
                modul_filter_assign[p[0].error_connection],
                p[1]
            ]

            # Add to PES items
            pes_items.extend(data)

        # Generate all the regions in turn, then return a new vertex
        # instance.
        encoders_with_gain = ens.encoders * ens.gains[:, np.newaxis]
        bias_with_di = np.dot(encoders_with_gain, ens.direct_input) + ens.bias

        system_region = utils.vertices.UnpartitionedListRegion(
            system_items, n_atoms_index=2)
        bias_region = utils.vertices.MatrixRegionPartitionedByRows(
            bias_with_di, formatter=utils.fp.bitsk)
        encoders_region = utils.vertices.MatrixRegionPartitionedByRows(
            encoders_with_gain, formatter=utils.fp.bitsk)
        decoders_region = utils.vertices.MatrixRegionPartitionedByRows(
            ens.decoders, formatter=utils.fp.bitsk)
        output_keys_region = utils.vertices.UnpartitionedKeysRegion(
            ens.output_keyspaces)
        gain_region = utils.vertices.MatrixRegionPartitionedByRows(
            ens.gains, formatter=utils.fp.bitsk)
        pes_region = utils.vertices.UnpartitionedListRegion(pes_items)
        spikes_region = utils.vertices.BitfieldBasedRecordingRegion(
            assembler.n_ticks)

        vertex = cls(ens.n_neurons, system_region, bias_region,
                     encoders_region, decoders_region, output_keys_region,
                     input_filter_region, input_filter_routing,
                     inhib_filter_region, inhib_filter_routing, gain_region,
                     modul_filter_region, modul_filter_routing,
                     pes_region, spikes_region)
        vertex.probes = ens.probes
        return vertex
