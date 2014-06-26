import numpy as np

import nengo
import nengo.builder
import nengo.decoders
from nengo.utils import distributions as dists
from nengo.utils.compat import is_integer
from nengo.utils.inspect import checked_call

from .utils import connections, fp, filters, vertices
from . import utils


@filters.with_filters(6, 7)
class EnsembleVertex(vertices.NengoVertex):
    """PACMAN Vertex for an Ensemble."""
    REGIONS = vertices.ordered_regions('SYSTEM', 'BIAS', 'ENCODERS',
                                       'DECODERS', 'OUTPUT_KEYS',
                                       **{'INHIB_FILTER': 8,
                                          'INHIB_ROUTING': 9,
                                          'GAIN': 10,
                                          'SPIKES': 15})
    MODEL_NAME = "nengo_ensemble"

    def __init__(self, ens, rng, dt=0.001, time_step=1000, constraints=None):
        """Create a new EnsembleVertex using the given Ensemble to generate
        appropriate parameters.

        :param ens: Ensemble to represent in the simulation.
        :param rng: RandomState
        :param dt:
        :param time_step: Machine timestep (in microseconds)
        """
        # Save a reference to the ensemble before unpacking some useful values
        self._ens = ens
        self.dt = dt
        self.time_step = time_step
        self.record_spikes = False

        # Create random number generator
        if ens.seed is None:
            rng = np.random.RandomState(rng.tomaxint())
        else:
            rng = np.random.RandomState(ens.seed)
        self.rng = rng

        # Generate eval points
        if ens.eval_points is None:
            dims, neurons = ens.dimensions, ens.n_neurons
            n_points = max(np.clip(500 * dims, 750, 2500), 2 * neurons)
            self.eval_points = dists.UniformHypersphere(ens.dimensions).sample(
                n_points, rng=rng) * ens.radius
        elif is_integer(ens.eval_points):
            self.eval_points = dists.UniformHypersphere(ens.dimensions).sample(
                ens.eval_points, rng=rng) * ens.radius
        else:
            self.eval_points = np.array(ens.eval_points, dtype=np.float64)
            if self.eval_points.ndim == 1:
                self.eval_points.shape = (-1, 1)

        # Set up neurons
        gain = ens.gain
        bias = ens.bias
        if gain is None or bias is None:
            # if max_rates and intercepts are distributions,
            # turn them into fixed samples.
            if hasattr(ens.max_rates, 'sample'):
                ens.max_rates = ens.max_rates.sample(
                    ens.n_neurons, rng=rng
                )
            if hasattr(ens.intercepts, 'sample'):
                ens.intercepts = ens.intercepts.sample(
                    ens.n_neurons, rng=rng
                )
            (gain, bias) = ens.neuron_type.gain_bias(
                ens.max_rates, ens.intercepts)

        self.bias = bias
        self.gain = gain
        self.tau_rc = ens.neuron_type.tau_rc
        self.tau_ref = ens.neuron_type.tau_ref

        self.n_input_dimensions = ens.dimensions

        # Set up encoders
        if ens.encoders is None:
            if isinstance(ens.neurons, nengo.Direct):
                self.encoders = np.identity(ens.dimensions)
            else:
                sphere = dists.UniformHypersphere(
                    ens.dimensions, surface=True)
                self.encoders = sphere.sample(
                    ens.n_neurons, rng=self.rng)
        else:
            self.encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.n_neurons, ens.dimensions)
            if self.encoders.shape != enc_shape:
                raise nengo.builder.ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions);"
                    " in this case %s." % (self.encoders.shape, enc_shape)
                )

            norm = np.sum(self.encoders ** 2, axis=1)[:, np.newaxis]
            self.encoders /= np.sqrt(norm)

        self.encoders_with_gain = self.encoders * self.gain[:, None]

        # Inhibition
        self.inhibitory_edge = None

        # For constant value injection
        self.direct_input = np.zeros(self._ens.dimensions)

        # Create the vertex
        super(EnsembleVertex, self).__init__(
            self._ens.n_neurons, constraints=constraints, label=ens.label
        )

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_region_system(self, n_atoms):
        return 8

    @vertices.region_pre_prepare('BIAS')
    def preprepare_region_bias(self):
        # Add the direct input to the bias current
        self.bias_with_di = (
            self.bias + np.dot(self.encoders_with_gain, self.direct_input)
        )

    @vertices.region_pre_sizeof('BIAS')
    def sizeof_region_bias(self, n_atoms):
        return n_atoms

    @vertices.region_pre_sizeof('ENCODERS')
    def sizeof_region_encoders(self, n_atoms):
        return n_atoms * self.n_input_dimensions

    @vertices.region_pre_prepare('DECODERS')
    def prepare_region_decoders(self):
        """Generate decoders for the Ensemble."""
        # Get a list of unique transform/function/solver triples, the width and
        # connection indices of this list.
        tfses = connections.ConnectionsWithSolvers(
            [edge.conn for edge in self.out_edges]
        )
        self._edge_decoders = dict([(edge, tfses[edge.conn]) for edge in
                                    self.out_edges])

        # Generate each decoder in turn
        decoders = list()
        decoder_builder = utils.decoders.DecoderBuilder(self._build_decoder)
        for tfse in tfses.transforms_functions:
            decoders.append(decoder_builder.get_transformed_decoder(
                tfse.function, tfse.transform, tfse.eval_points, tfse.solver
            ))

        # Compress and merge the decoders
        # @neworderofjamie -- modify the True as required for learnt decoders
        (self.decoder_headers, self._merged_decoders) = \
            utils.decoders.get_combined_compressed_decoders(
                decoders, compress=[True for d in decoders])

        self._merged_decoders /= self.dt
        self.n_output_dimensions = len(self.decoder_headers)

    def _build_decoder(self, function, eval_points, solver):
        if eval_points is None:
            eval_points = self.eval_points

        x = np.dot(eval_points, self.encoders.T / self._ens.radius)
        activities = self._ens.neuron_type.rates(x, self.gain, self.bias)

        if function is None:
            targets = eval_points
        else:
            (value, invoked) = checked_call(function, eval_points[0])
            function_size = np.asarray(value).size
            targets = np.zeros((len(eval_points), function_size))
            for (i, ep) in enumerate(eval_points):
                targets[i] = function(ep)

        if solver is None:
            solver = nengo.decoders.LstsqL2()

        decoder = solver(activities, targets, self.rng)

        if isinstance(decoder, tuple):
            decoder = decoder[0]
        return decoder

    @vertices.region_pre_sizeof('DECODERS')
    def sizeof_region_decoders(self, n_atoms):
        return n_atoms * self.n_output_dimensions

    @vertices.region_pre_sizeof('OUTPUT_KEYS')
    def sizeof_region_output_keys(self, n_atoms):
        return self.n_output_dimensions

    @vertices.region_pre_prepare('INHIB_FILTER')
    def prepare_region_inhib_filters(self):
        self.inhib_dims = (0 if self.inhibitory_edge is None else
                           self.inhibitory_edge.transform.size)
        self.inhib_gain = (0 if self.inhibitory_edge is None else
                           self.inhibitory_edge.transform)

    @vertices.region_pre_sizeof('INHIB_FILTER')
    def pre_sizeof_region_inhib_filters(self, n_atoms):
        return 1 + 3

    @vertices.region_pre_sizeof('INHIB_ROUTING')
    def pre_sizeof_region_inhib_routing(self, n_atoms):
        return 4 * 5

    @vertices.region_pre_sizeof('GAIN')
    def pre_sizeof_region_inhib_gain(self, n_atoms):
        return n_atoms

    @vertices.region_post_prepare('INHIB_ROUTING')
    def post_prepare_inhib_routing(self):
        self.inhib_filter_keys = list()

        if self.inhibitory_edge is not None:
            kms = [(subedge.edge.prevertex.generate_routing_info(subedge),
                    subedge.edge.dimension_mask) for subedge in
                   self.inhibitory_edge.subedges]

            self.inhib_filter_keys.extend(
                [filters.FilterRoute(km[0], km[1], 0, dm) for (km, dm) in
                 kms])

    @vertices.region_sizeof('INHIB_ROUTING')
    def sizeof_region_inhib_routing(self, subvertex):
        return 4 * len(self.inhib_filter_keys) + 1

    @vertices.region_pre_sizeof('SPIKES')
    def sizeof_region_recording(self, n_atoms):
        size = 0
        if self.record_spikes and self.runtime is not None:
            frame_length = (n_atoms >> 5) + (1 if n_atoms & 0x1f else 0)
            n_frames = int(self.runtime * 1000)  # TODO timestep scaling
            size = n_frames * frame_length
        return size + 1

    def cpu_usage(self, n_atoms):
        """Return the CPU utilisation for the specified atoms."""
        # TODO: Calculate this
        return 0

    def dtcm_usage(self, n_atoms):
        """The recording region is not copied into DTCM."""
        size = sum([r.pre_sizeof(n_atoms) for r in self._regions])
        size -= self.sizeof_region_recording(n_atoms)
        return size*4

    def get_maximum_atoms_per_core(self):
        # TODO: Calculate this
        return 128

    @vertices.region_write('SYSTEM')
    def write_region_system(self, subvertex, spec):
        """Write the system region for the given subvertex."""
        spec.write(data=self.n_input_dimensions)
        spec.write(data=self.n_output_dimensions)
        spec.write(data=subvertex.n_atoms)
        spec.write(data=self.time_step)
        spec.write(data=int(self.tau_ref / (self.time_step * 10**-6)))
        spec.write(data=fp.bitsk(self.dt / self.tau_rc))
        spec.write(data=0x1 if self.record_spikes else 0x0)  # Recording flag
        spec.write(data=self.inhib_dims)

    @vertices.region_write('BIAS')
    def write_region_bias(self, subvertex, spec):
        """Write the bias region for the given subvertex."""
        spec.write_array(fp.bitsk(
            self.bias_with_di[subvertex.lo_atom:subvertex.hi_atom+1]))

    @vertices.region_write('ENCODERS')
    def write_region_encoders(self, subvertex, spec):
        """Write the encoder region for the given subvertex."""
        for n in range(subvertex.lo_atom, subvertex.hi_atom + 1):
            for d in range(self.n_input_dimensions):
                spec.write(data=fp.bitsk(self.encoders_with_gain[n, d]))

    @vertices.region_write('DECODERS')
    def write_region_decoders(self, subvertex, spec):
        """Write the decoder region for the given subvertex."""
        for n in range(subvertex.lo_atom, subvertex.hi_atom + 1):
            # Write the decoders for all the atoms within this subvertex
            for d in range(self.n_output_dimensions):
                spec.write(data=fp.bitsk(self._merged_decoders[n][d]))

    @vertices.region_write('OUTPUT_KEYS')
    def write_region_output_keys(self, subvertex, spec):
        """Write the output keys region for the given subvertex."""
        x, y, p = subvertex.placement.processor.get_coordinates()

        for (h, i, d) in self.decoder_headers:
            # Generate the routing keys for each dimension
            # TODO Use KeySpaces to perform this calculation
            spec.write(data=((x << 24) | (y << 16) | ((p-1) << 11) |
                             (i << 6) | d))

    @vertices.region_write('INHIB_FILTER')
    def write_region_inhib_filter(self, subvertex, spec):
        if self.inhibitory_edge is not None:
            spec.write(data=1)
            f = (np.exp(-self.dt / self.inhibitory_edge.synapse) if
                 self.inhibitory_edge.synapse is not None else 0.)
            spec.write(data=fp.bitsk(f))
            spec.write(data=fp.bitsk(1 - f))
            spec.write(
                data=(0x0 if self.inhibitory_edge._filter_is_accumulatory else
                      0xffffffff))
        else:
            spec.write(data=0)

    @vertices.region_write('INHIB_ROUTING')
    def write_region_inhib_routing(self, subvertex, spec):
        routes = self.inhib_filter_keys

        spec.write(data=len(routes))
        for route in routes:
            spec.write(data=route.key)
            spec.write(data=route.mask)
            spec.write(data=route.index)
            spec.write(data=route.dimension_mask)

    @vertices.region_write('GAIN')
    def write_region_inhib_gain(self, subvertex, spec):
        gains = fp.bitsk(self.gain[subvertex.lo_atom:subvertex.hi_atom+1])
        spec.write_array(gains)

    def generate_routing_info(self, subedge):
        """Generate a key and mask for the given subedge."""
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        i = self._edge_decoders[subedge.edge]

        return subedge.edge.generate_key(x, y, p, i), subedge.edge.mask
