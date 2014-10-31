import numpy as np

import nengo
from nengo.builder import sample as builder_sample
from nengo.utils import distributions as dists
import nengo.utils.numpy as npext
from nengo.utils.stdlib import checked_call

from . import connections as ens_conn_utils
from . import decoders as decoder_utils
from . import intermediate
from . import pes as pes_utils

from ..utils.fixpoint import bitsk
from ..utils import filters as filter_utils
from ..utils import regions as region_utils

from ..spinnaker import regions, vertices


class IntermediateLIF(intermediate.IntermediateEnsemble):
    def __init__(self, n_neurons, gains, bias, encoders, decoders, tau_rc,
                 tau_ref, eval_points, decoder_headers, learning_rules):
        super(IntermediateLIF, self).__init__(
            n_neurons, gains, bias, encoders, decoders, eval_points,
            decoder_headers, learning_rules)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    @classmethod
    def build(cls, ensemble, connection_trees, config, rngs, direct_input,
              record_spikes):
        """Build a single LIF ensemble into an intermediate form.
        """
        # Get the rng for this object
        rng = rngs[ensemble]

        # Determine max_rates and intercepts
        max_rates = builder_sample(ensemble.max_rates, ensemble.n_neurons, rng)
        intercepts = builder_sample(ensemble.intercepts, ensemble.n_neurons,
                                    rng)

        # Generate gains, bias
        gain, bias = ensemble.neuron_type.gain_bias(max_rates, intercepts)

        # Generate encoders
        if isinstance(ensemble.encoders, dists.Distribution):
            encoders = ensemble.encoders.sample(
                ensemble.n_neurons, ensemble.dimensions, rng=rng)
        else:
            encoders = npext.array(ensemble.encoders, min_dims=2,
                                   dtype=np.float64)
            encoders /= npext.norm(encoders, axis=1, keepdims=True)

        # Generate decoders for outgoing connections
        decoders = list()

        def build_decoder(function, evals, solver):
            """Internal function for building a single decoder."""
            assert solver is None or not solver.weights

            x = np.dot(evals, encoders.T / ensemble.radius)
            activities = ensemble.neuron_type.rates(x, gain, bias)

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
        decoders_to_compress = list()
        for c in connection_trees.get_outgoing_connections(ensemble):
            # Build the decoder
            decoders.append(
                decoder_builder.get_transformed_decoder(
                    c.function, c.transform.T, c.eval_points, c.solver
                )
            )

            # Keep track of which decoders we can compress
            decoders_to_compress.append(
                True if c.transmitter_learning_rule is None else False)

        # Compress and merge the decoders
        (decoder_headers, decoders) =\
            decoder_utils.get_combined_compressed_decoders(
                decoders, compress=decoders_to_compress)

        return cls(ensemble.n_neurons, gain, bias, encoders, decoders,
                   ensemble.neuron_type.tau_rc, ensemble.neuron_type.tau_ref,
                   ensemble.eval_points, decoder_headers)


class SystemRegion(regions.Region):
    """Region representing parameters for a LIF ensemble.
    """
    def __init__(self, n_input_dimensions, n_output_dimensions,
                 machine_timestep, t_ref, dt_over_t_rc, record_spikes):
        """Create a new system region for a LIF ensemble.

        :param n_input_dimensions: Number of input dimensions (TODO: remove)
        :param n_output_dimensions: Number of output dimensions (TODO: remove)
        :param machine_timestep: Timestep in microseconds (TODO: remove)
        :param float t_ref: Refractory period.
        :param float dt_over_t_rc:
        :param bool: Record spikes (TODO: Move elsewhere?)
        """
        self.n_input_dimensions = n_input_dimensions
        self.n_output_dimensions = n_output_dimensions
        self.machine_timestep = machine_timestep
        self.t_ref_in_ticks = int(t_ref / (machine_timestep * 10**-6))
        self.dt_over_t_rc = bitsk(dt_over_t_rc)
        self.record_flags = 0x1 if record_spikes else 0x0

    def sizeof(self, vertex_slice):
        """Get the size of this region in WORDS."""
        # Is a constant, 8
        return 8

    def create_subregion(self, vertex_slice, vertex_index):
        """Create a subregion for this slice of the vertex.
        """
        data = np.array([
            self.n_input_dimensions,
            self.n_output_dimensions,
            vertex_slice.stop - vertex_slice.start,
            self.machine_timestep,
            self.t_ref_in_ticks,
            self.dt_over_t_rc,
            self.record_flags,
            1,
        ], dtype=np.uint32)

        return regions.Subregion(data, len(data), False)


class EnsembleLIF(vertices.Vertex):
    executable_path = None  # TODO

    def __init__(self, n_neurons, system_region, bias_region, encoders_region,
                 decoders_region, output_keys_region, input_filter_region,
                 input_filter_routing, inhib_filter_region,
                 inhib_filter_routing, gain_region, modulatory_filter_region,
                 modulatory_filter_routing, pes_region, spikes_region):
        regions = [
            system_region,  # 1
            bias_region,  # 2
            encoders_region,  # 3
            decoders_region,  # 4
            output_keys_region,  # 5
            input_filter_region,  # 6
            input_filter_routing,  # 7
            inhib_filter_region,  # 8
            inhib_filter_routing,  # 9
            gain_region,  # 10
            modulatory_filter_region,  # 11
            modulatory_filter_routing,  # 12
            pes_region,  # 13
            None,  # 14
            spikes_region,  # 15
        ]
        super(EnsembleLIF, self).__init__(n_neurons, '', regions)

    @classmethod
    def assemble_from_intermediate(cls, ens, assembler):
        # Create a system region
        system_region = SystemRegion(
            n_input_dimensions=ens.n_dimensions,
            n_output_dimensions=len(ens.output_keys),
            machine_timestep=assembler.timestep,
            t_ref=ens.tau_ref,
            dt_over_t_rc=assembler.dt / ens.tau_rc,
            record_spikes=ens.record_spikes
        )

        # Prepare the input filtering regions
        # Prepare the inhibitory filtering regions
        in_conns = assembler.get_incoming_connections(ens)
        inhib_conns =\
            [c for c in in_conns if isinstance(
                c, ens_conn_utils.IntermediateGlobalInhibitionConnection)]
        modul_conns = [c for c in in_conns if c.modulatory]
        input_conns = [c for c in in_conns
                       if c not in inhib_conns and c not in modul_conns]

        (input_filter_region, input_filter_routing) =\
            filter_utils.get_filter_regions(input_conns, assembler.dt)
        (inhib_filter_region, inhib_filter_routing) =\
            filter_utils.get_filter_regions(inhib_conns, assembler.dt)
        (modul_filter_region, modul_filter_routing) =\
            filter_utils.get_filter_regions(modul_conns, assembler.dt)
        _, modul_filter_assign = filter_utils.get_combined_filters(modul_conns)

        # Assert that only known learning rules are present
        _learning_types = set(l.rule.__class__ for l in ens.learning_rules)
        _unsupported = _learning_types - set([nengo.PES])
        if len(_unsupported) > 0:
            raise NotImplementedError(
                'Encountered unsupported learning rule types {:s}'.format(
                    ', '.join(l.__name__ for l in _unsupported)
                )
            )

        # Create the PES region
        pes_region = pes_utils.make_pes_region(
            ens.learning_rules, assembler.dt, modul_filter_assign)

        # Generate all the regions in turn, then return a new vertex
        # instance.
        encoders_with_gain = ens.encoders * ens.gains[:, np.newaxis]
        bias_with_di = np.dot(encoders_with_gain, ens.direct_input) + ens.bias

        bias_region = regions.MatrixRegionPartitionedByRows(
            bias_with_di, formatter=bitsk)
        encoders_region = regions.MatrixRegionPartitionedByRows(
            encoders_with_gain, formatter=bitsk)
        decoders_region = regions.MatrixRegionPartitionedByRows(
            ens.decoders, formatter=bitsk)
        output_keys_region = regions.KeysRegion(ens.output_keys)
        gain_region = regions.MatrixRegionPartitionedByRows(
            ens.gains, formatter=bitsk)
        spikes_region = region_utils.BitfieldBasedRecordingRegion(
            assembler.n_ticks)

        vertex = cls(ens.n_neurons, system_region, bias_region,
                     encoders_region, decoders_region, output_keys_region,
                     input_filter_region, input_filter_routing,
                     inhib_filter_region, inhib_filter_routing, gain_region,
                     modul_filter_region, modul_filter_routing,
                     pes_region, spikes_region)
        vertex.probes = ens.probes
        return vertex
