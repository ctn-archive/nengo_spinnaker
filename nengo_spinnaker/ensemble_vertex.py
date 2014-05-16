import numpy as np

import nengo
import nengo.builder
import nengo.decoders
from nengo.utils import distributions as dists
from nengo.utils.compat import is_integer

from .utils import bins, fp, filters, vertices


@filters.with_filters(6, 7)
class EnsembleVertex(vertices.NengoVertex):
    """PACMAN Vertex for an Ensemble."""
    REGIONS = vertices.ordered_regions('SYSTEM', 'BIAS', 'ENCODERS',
                                       'DECODERS', 'OUTPUT_KEYS')
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

        # Create random number generator
        if ens.seed is None:
            rng = np.random.RandomState(rng.tomaxint())
        else:
            rng = np.random.RandomState(ens.seed)
        self.rng = rng

        # Generate eval points
        if ens.eval_points is None:
            dims, neurons = ens.dimensions, ens.neurons.n_neurons
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
        gain = ens.neurons.gain
        bias = ens.neurons.bias
        if gain is None or bias is None:
            # if max_rates and intercepts are distributions,
            # turn them into fixed samples.
            if hasattr(ens.max_rates, 'sample'):
                ens.max_rates = ens.max_rates.sample(
                    ens.neurons.n_neurons, rng=rng
                )
            if hasattr(ens.intercepts, 'sample'):
                ens.intercepts = ens.intercepts.sample(
                    ens.neurons.n_neurons, rng=rng
                )
            (gain, bias) = ens.neurons.gain_bias(ens.max_rates, ens.intercepts)

        self.bias = bias
        self.gain = gain
        self.tau_rc = ens.neurons.tau_rc
        self.tau_ref = ens.neurons.tau_ref

        self.n_input_dimensions = ens.dimensions

        # Set up encoders
        if ens.encoders is None:
            if isinstance(ens.neurons, nengo.Direct):
                self.encoders = np.identity(ens.dimensions)
            else:
                sphere = dists.UniformHypersphere(
                    ens.dimensions, surface=True)
                self.encoders = sphere.sample(
                    ens.neurons.n_neurons, rng=self.rng)
        else:
            self.encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
            if self.encoders.shape != enc_shape:
                raise nengo.builder.ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions);"
                    " in this case %s." % (self.encoders.shape, enc_shape)
                )

            norm = np.sum(self.encoders ** 2, axis=1)[:, np.newaxis]
            self.encoders /= np.sqrt(norm)

        self.encoders_with_gain = self.encoders * self.gain[:, None]

        # TODO: remove this when it is not required be Ensemble.activities()
        ens.encoders = self.encoders

        # For constant value injection
        self.direct_input = np.zeros(self._ens.dimensions)

        # Decoders and Filters
        self.decoders = bins.DecoderBin(rng)

        # Create the vertex
        super(EnsembleVertex, self).__init__(
            self._ens.n_neurons, constraints=constraints, label=ens.label
        )

    @property
    def _tau_ref_in_steps(self):
        return self.tau_ref / (self.time_step * 10**-6)

    @property
    def _dt_over_tau_rc(self):
        return self.dt / self.tau_rc

    @property
    def n_output_dimensions(self):
        """The sum of the decoders in the decoder bin."""
        return self.decoders.width

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_region_system(self, n_atoms):
        return 7

    @vertices.region_pre_sizeof('BIAS')
    def sizeof_region_bias(self, n_atoms):
        return n_atoms

    @vertices.region_pre_sizeof('ENCODERS')
    def sizeof_region_encoders(self, n_atoms):
        return n_atoms * self.n_input_dimensions

    @vertices.region_pre_sizeof('DECODERS')
    def sizeof_region_decoders(self, n_atoms):
        return n_atoms * self.n_output_dimensions

    @vertices.region_pre_sizeof('OUTPUT_KEYS')
    def sizeof_region_output_keys(self, n_atoms):
        return self.n_output_dimensions

    def cpu_usage(self, n_atoms):
        """Return the CPU utilisation for the specified atoms."""
        # TODO: Calculate this
        return 0

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
        spec.write(data=self._tau_ref_in_steps)
        spec.write(data=fp.bitsk(self._dt_over_tau_rc))

    @vertices.region_pre_prepare('BIAS')
    def preprepare_region_bias(self):
        # Add the direct input to the bias current
        self.bias += np.dot(self.encoders_with_gain, self.direct_input)

    @vertices.region_write('BIAS')
    def write_region_bias(self, subvertex, spec):
        """Write the bias region for the given subvertex."""
        spec.write_array(fp.bitsk(
            self.bias[subvertex.lo_atom:subvertex.hi_atom+1]))

    @vertices.region_write('ENCODERS')
    def write_region_encoders(self, subvertex, spec):
        """Write the encoder region for the given subvertex."""
        for n in range(subvertex.lo_atom, subvertex.hi_atom + 1):
            for d in range(self.n_input_dimensions):
                spec.write(data=fp.bitsk(self.encoders_with_gain[n, d]))

    @vertices.region_write('DECODERS')
    def write_region_decoders(self, subvertex, spec):
        """Write the decoder region for the given subvertex."""
        decoders = self.decoders.get_merged_decoders()

        for n in range(subvertex.lo_atom, subvertex.hi_atom + 1):
            # Write the decoders for all the atoms within this subvertex
            for d in range(self.n_output_dimensions):
                spec.write(data=fp.bitsk(decoders[n][d] / self.dt))

    @vertices.region_write('OUTPUT_KEYS')
    def write_region_output_keys(self, subvertex, spec):
        """Write the output keys region for the given subvertex."""
        x, y, p = subvertex.placement.processor.get_coordinates()

        for (i, w) in enumerate(self.decoders.decoder_widths):
            # Generate the routing keys for each dimension
            for d in range(w):
                spec.write(data=((x << 24) | (y << 16) | ((p-1) << 11) |
                                 (i << 6) | d))

    def generate_routing_info(self, subedge):
        """Generate a key and mask for the given subedge."""
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        i = self.decoders.edge_index(subedge.edge)
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

        return key, 0xFFFFFFE0
