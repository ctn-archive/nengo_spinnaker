import math
import numpy as np

from nengo.builder.ensemble import sample as builder_sample
from nengo.utils import distributions as dists
import nengo.utils.numpy as npext

from . import decoders as decoder_utils
from . import intermediate
from . import pes as pes_utils

from ..assembler import Assembler
from ..connections.reduced import StandardInputPort, GlobalInhibitionPort
from ..utils.fixpoint import bitsk
from ..utils import filters as filter_utils
from ..utils import regions as region_utils

from ..spinnaker import regions, vertices


class IntermediateLIF(intermediate.IntermediateEnsemble):
    """Intermediate representation of an ensemble of LIF neurons."""
    def __init__(self, n_neurons, gains, bias, encoders, decoders, tau_rc,
                 tau_ref, decoder_headers, learning_rules, direct_input):
        super(IntermediateLIF, self).__init__(
            n_neurons, gains, bias, encoders, decoders, decoder_headers,
            learning_rules, direct_input)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    @classmethod
    def build(cls, placeholder, connection_trees, config, rng):
        """Build a single LIF ensemble into an intermediate form.
        """
        ensemble = placeholder.ens
        direct_input = placeholder.direct_input

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
        decoder_headers = list()
        decoder_compress = list()
        learning_rules = list()

        decoder_builder = decoder_utils.create_decoder_builder(
            encoders=encoders, radius=ensemble.radius, gain=gain, bias=bias,
            rates=ensemble.neuron_type.rates, rng=rng
        )

        # Build each of the decoders in turn
        offset = 0
        for c in connection_trees.get_outgoing_connections(placeholder):
            # Build the decoder
            decoder, solver_info, headers = decoder_builder(c)

            # Store which decoders we can compress, currently none, this is
            # because of learning rules.  Also, some compression already occurs
            # as we know the post-slicing.
            decoder_compress.append(False)

            # Store
            decoders.append(decoder)
            decoder_headers.extend(headers)

            # If this is a PES connection then store the offset to the decoder
            if isinstance(c.transmitter_learning_rule, pes_utils.PESInstance):
                ilr = intermediate.IntermediateLearningRule(
                    c.transmitter_learning_rule, offset)
                learning_rules.append(ilr)
            offset += decoder.shape[1]  # Update the column offset

        # Combine and compress decoders
        decoder_headers, decoders = \
            decoder_utils.get_combined_compressed_decoders(
                decoders, decoder_headers, compress=decoder_compress)

        return cls(ensemble.n_neurons, gain, bias, encoders, decoders,
                   ensemble.neuron_type.tau_rc, ensemble.neuron_type.tau_ref,
                   decoder_headers, learning_rules, direct_input)


class EnsembleLIF(vertices.Vertex):
    executable_path = None  # TODO

    def __init__(self, n_neurons, system_region, bias_region, encoders_region,
                 decoders_region, output_keys_region, input_filter_region,
                 input_filter_routing, inhib_filter_region,
                 inhib_filter_routing, gain_region, modulatory_filter_region,
                 modulatory_filter_routing, pes_region, spikes_region):
        """
        :type system_region: SystemRegion
        """
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
        self.n_input_dimensions = system_region.n_input_dimensions
        self.n_output_dimension = system_region.n_output_dimensions
        super(EnsembleLIF, self).__init__(n_neurons, '', regions)

    def get_cpu_usage_for_atoms(self, vertex_slice):
        """Get the CPU requirements in cycles for this vertex slice."""
        # TODO Determine what this is...
        return 0

    def get_dtcm_usage_static(self, vertex_slice):
        """Get the memory requirement (in WORDS) not arising from data already
        specified in regions.

        For each input dimension we store: 1 word current input
        For each output dimension we store: 1 word of buffer
        For each neuron we store: 1 word of state

        .. todo::
            Add any missing elements of memory usage.

        Parameters
        ----------
        vertex_slice : :py:func:`slice`
            Slice representing the atoms selected for this partition of the
            vertex.
        """
        return (self.n_input_dimensions + self.n_output_dimension +
                min(self.n_atoms, vertex_slice.stop - vertex_slice.start))


@Assembler.object_assembler(IntermediateLIF)
def assemble_lif_vertex_from_intermediate(obj, connection_trees, config, rngs,
                                          runtime, dt, machine_timestep):
    """Convert an intermediate LIF population into a vertex.

    Parameters
    ----------
    obj : IntermediateLIF
        Intermediate representation of a LIF ensemble.
    connection_trees : ..connections.connection_tree.ConnectionTree
        Connectivity of the model.
    config : ..config.Config
        Configuration.
    rngs : :py:func:`dict`
        A mapping of objects to random number generators.
    runtime : float
        The runtime of the simulation in seconds.
    dt : float
        The duration of a simulation step in seconds.
    machine_timestep : int
        Real-time duration of a simulation step in microseconds.

    Returns
    -------
    EnsembleLIF
        A vertex representing an ensemble of LIF neurons to simulate on
        SpiNNaker.
    """
    # Construct the system region for the Ensemble
    system_region = SystemRegion(
        obj.size_in, len(obj.decoder_headers), machine_timestep, obj.tau_ref,
        dt / obj.tau_rc, obj.record_spikes)

    # Construct the various filter and filter routing regions
    in_conns = connection_trees.get_incoming_connections(obj)
    input_filter, input_routing = filter_utils.get_filter_regions(
        in_conns[StandardInputPort], dt, obj.size_in)
    inhib_filter, inhib_routing = filter_utils.get_filter_regions(
        in_conns[GlobalInhibitionPort], dt, 1)

    # Get the filters and routing for PES connections
    pes_region, pes_filter_region, pes_filter_routing_region = \
        pes_utils.make_pes_regions(obj.learning_rules, in_conns, dt)

    # Construct the regions for encoders and decoders.
    # TODO Interleave these and page through them in the C code.
    encoders_with_gain = obj.encoders * obj.gains[:, np.newaxis]
    encoder_region = regions.MatrixRegionPartitionedByRows(encoders_with_gain,
                                                           formatter=bitsk)
    decoder_region = regions.MatrixRegionPartitionedByRows(obj.decoders,
                                                           formatter=bitsk)

    # Construct the output keys region from the decoder headers
    output_keys_region = regions.KeysRegion(obj.decoder_headers,
                                            fill_in_field='s')

    # Construct the gain and bias regions
    gain_region = regions.MatrixRegionPartitionedByRows(obj.gains,
                                                        formatter=bitsk)
    bias_with_di = encoders_with_gain.dot(obj.direct_input) + obj.bias
    bias_region = regions.MatrixRegionPartitionedByRows(bias_with_di,
                                                        formatter=bitsk)

    # Create the spikes recording region
    n_ticks = int(math.ceil(runtime / dt))
    spikes_region = region_utils.BitfieldBasedRecordingRegion(n_ticks)

    # Build the vertex and return
    return EnsembleLIF(
        obj.n_neurons, system_region, bias_region, encoder_region,
        decoder_region, output_keys_region, input_filter, input_routing,
        inhib_filter, inhib_routing, gain_region, pes_filter_region,
        pes_filter_routing_region, pes_region, spikes_region
    )


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
        super(SystemRegion, self).__init__()

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
