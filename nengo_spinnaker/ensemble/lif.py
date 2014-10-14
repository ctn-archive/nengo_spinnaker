import numpy as np
import warnings

import nengo
from nengo.utils import distributions as dists
import nengo.utils.builder as builder_utils
import nengo.utils.numpy as npext
from nengo.utils.stdlib import checked_call

from . import connections as ens_conn_utils
from . import decoders as decoder_utils
from . import intermediate


class IntermediateLIF(intermediate.IntermediateEnsemble):
    def __init__(self, n_neurons, gains, bias, encoders, decoders, tau_rc,
                 tau_ref, eval_points, decoder_headers, learning_rules):
        super(IntermediateLIF, self).__init__(
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
                n_points = builder_utils.default_n_eval_points(ens.n_neurons,
                                                               ens.dimensions)
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
        tfses = ens_conn_utils.get_combined_outgoing_ensemble_connections(
            out_conns)

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

        decoder_builder = decoder_utils.DecoderBuilder(build_decoder)

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
            decoder_utils.get_combined_compressed_decoders(
                decoders, compress=decoders_to_compress)
        decoders /= dt

        return cls(ens.n_neurons, gain, bias, encoders, decoders,
                   ens.neuron_type.tau_rc, ens.neuron_type.tau_ref,
                   eval_points, decoder_headers, learning_rules)


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
            [c for c in in_conns if isinstance(
                c, ens_conn_utils.IntermediateGlobalInhibitionConnection)]
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


assert False, "Incomplete!"
class SystemRegion(object):
    pass
