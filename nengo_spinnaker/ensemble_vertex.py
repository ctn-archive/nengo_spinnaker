import os
import numpy as np

import nengo
import nengo.builder
import nengo.decoders
from nengo.utils import distributions
from pacman103.lib import graph, data_spec_gen, lib_map, parameters
from pacman103.front.common import enums

from . import collections


class EnsembleVertex(graph.Vertex):
    """PACMAN Vertex for an Ensemble."""

    REGIONS = enums.enum1(
        'SYSTEM',
        'BIAS',
        'ENCODERS',
        'DECODERS',
        'OUTPUT_KEYS'
    )

    model_name = "nengo_ensemble"

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
            self.eval_points = distributions.UniformHypersphere(
                ens.dimensions).sample(ens.EVAL_POINTS, rng=rng) * ens.radius
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
                sphere = distributions.UniformHypersphere(
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
        ens.encoders = self.encoders   # TODO: remove this when it is no longer
                                       # required be Ensemble.activities()

        # For constant value injection
        self.direct_input = np.zeros(self._ens.dimensions)

        # Decoders and Filters
        self.decoders = collections.DecoderBin(rng)
        self.filters = collections.FilterCollection()

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

    def sizeof_region_system(self):
        """Get the size (in bytes) of the SYSTEM region."""
        # 8 words, 4 bytes per word
        return 4 * 8

    def sizeof_region_bias(self, n_atoms):
        """Get the size (in bytes) of the BIAS region for the given number of
        neurons/atoms.
        """
        # 1 word per atom
        return 4 * n_atoms

    def sizeof_region_encoders(self, n_atoms):
        """Get the size (in bytes) of the ENCODERS region for the given number
        of neurons/atoms.
        """
        # 1 word per atom per input dimension
        return 4 * n_atoms * self.n_input_dimensions

    def sizeof_region_decoders(self, n_atoms):
        """Get the size (in bytes) of the DECODERS region for the given number
        of neurons/atoms.
        """
        # 1 word per atom per output dimension
        return 4 * n_atoms * self.n_output_dimensions

    def sizeof_region_output_keys(self):
        """Get the size (in bytes) of the OUTPUT_KEYS region."""
        # 1 word per output dimension
        return 4 * self.n_output_dimensions

    def sdram_usage(self, lo_atom, hi_atom):
        """Return the amount of SDRAM used for the specified atoms."""
        # At the moment this is the same as the DTCM usage, though this may
        # change.
        return self.dtcm_usage(lo_atom, hi_atom)

    def dtcm_usage(self, lo_atom, hi_atom):
        """Return the amount of DTCM used for the specified atoms."""
        n_atoms = hi_atom - lo_atom + 1
        return sum([
            self.sizeof_region_system(),
            self.sizeof_region_bias(n_atoms),
            self.sizeof_region_encoders(n_atoms),
            self.sizeof_region_decoders(n_atoms),
            self.sizeof_region_output_keys(),
        ])

    def cpu_usage(self, lo_atom, hi_atom):
        """Return the CPU utilisation for the specified atoms."""
        # TODO: Calculate this
        return 0

    def get_resources_for_atoms(self, lo_atom, hi_atom, n_machine_time_steps,
                                machine_time_step_us, partition_data_object):
        """Get the resources required for the specified atoms.

        :param lo_atom: Index of the lowest atom to represent
        :param hi_atom: Index of the highest atom to represent
        :param n_machine_time_steps: Duration of the simulation
        :param machine_time_step_us: Duration of a machine time step in us
        :param partition_data_object: ?

        :returns: A tuple of the partition data object, and the resources
                  required.
        """
        return lib_map.Resources(
            self.cpu_usage(lo_atom, hi_atom),
            self.dtcm_usage(lo_atom, hi_atom),
            self.sdram_usage(lo_atom, hi_atom)
        )

    def get_maximum_atoms_per_core(self):
        # TODO: Calculate this
        return 128

    def generateDataSpec(self, processor, subvertex, dao):
        """Generate the data spec for the given subvertex."""
        # Create a spec for the subvertex
        subvertex.spec = data_spec_gen.DataSpec(processor, dao)
        subvertex.spec.initialise(0xABCD, dao)
        subvertex.spec.comment("# Nengo Ensemble")

        # Finalise the values for this Ensemble
        # Encode any constant inputs, and add to the biases
        self.bias += np.dot(self.encoders, self.direct_input)

        # Generate the filters
        for e in self.in_edges:
            self.filters.add_edge(e)

        # Generate the list of decoders, and the list of ouput keys
        subvertex.output_keys = list()
        x, y, p = processor.get_coordinates()
        for (i, w) in enumerate(self.decoders.decoder_widths):
            # Generate the routing keys for each dimension
            for d in range(w):
                subvertex.output_keys.append(
                    (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6) | d
                )

        # Fill in the spec
        self.reserve_regions(subvertex)
        self.write_region_system(subvertex)
        self.write_region_bias(subvertex)
        self.write_region_encoders(subvertex)
        self.write_region_decoders(subvertex)
        self.write_region_output_keys(subvertex)

        # Close the spec
        subvertex.spec.endSpec()
        subvertex.spec.closeSpecFile()

        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            os.path.join(
                dao.get_common_binaries_directory(),
                'nengo_ensemble.aplx'
            ),
            x, y, p
        )

        return (executable_target, list(), list())

    def reserve_regions(self, subvertex):
        """Reserve sufficient space for the regions in the spec."""
        # TODO Modify the following functions to use write_array rather than
        #  lots of writes.
        subvertex.spec.reserveMemRegion(
            self.REGIONS.SYSTEM,
            self.sizeof_region_system()
        )
        subvertex.spec.reserveMemRegion(
            self.REGIONS.BIAS,
            self.sizeof_region_bias(subvertex.n_atoms)
        )
        subvertex.spec.reserveMemRegion(
            self.REGIONS.ENCODERS,
            self.sizeof_region_encoders(subvertex.n_atoms)
        )
        subvertex.spec.reserveMemRegion(
            self.REGIONS.DECODERS,
            self.sizeof_region_decoders(subvertex.n_atoms)
        )
        subvertex.spec.reserveMemRegion(
            self.REGIONS.OUTPUT_KEYS,
            self.sizeof_region_output_keys()
        )

    def write_region_system(self, subvertex):
        """Write the system region for the given subvertex."""
        subvertex.spec.switchWriteFocus(self.REGIONS.SYSTEM)
        subvertex.spec.comment("""# System Region
        # -------------
        # 1. Number of input dimensions
        # 2. Number of output dimensions
        # 3. Number of neurons
        # 4. Machine time step in us
        # 5. tau_ref in number of steps
        # 6. dt over tau_rc
        # 7. Filter decay (TO BE CHANGED)
        # 8. 1 - Filter decay (TO BE CHANGED)
        """)
        subvertex.spec.write(data=self.n_input_dimensions)
        subvertex.spec.write(data=self.n_output_dimensions)
        subvertex.spec.write(data=subvertex.n_atoms)
        subvertex.spec.write(data=self.time_step)
        subvertex.spec.write(data=self._tau_ref_in_steps)
        subvertex.spec.write(data=parameters.s1615(self._dt_over_tau_rc))

        if len(self.in_edges) > 0:
            filter = self.filters.filter_tcs(self.dt)
            subvertex.spec.write(data=parameters.s1615(filter[0][0]))
            subvertex.spec.write(data=parameters.s1615(filter[0][1]))

    def write_region_bias(self, subvertex):
        """Write the bias region for the given subvertex."""
        subvertex.spec.switchWriteFocus(self.REGIONS.BIAS)
        subvertex.spec.write_array(parameters.s1615(
            self.bias[subvertex.lo_atom:subvertex.hi_atom+1]))

    def write_region_encoders(self, subvertex):
        """Write the encoder region for the given subvertex."""
        subvertex.spec.switchWriteFocus(self.REGIONS.ENCODERS)
        subvertex.spec.comment("# Encoders Region")
        for n in range(subvertex.lo_atom, subvertex.hi_atom + 1):
            for d in range(self.n_input_dimensions):
                subvertex.spec.write(
                    data=parameters.s1615(
                        self.encoders[n, d] * self.gain[n]
                    )
                )

    def write_region_decoders(self, subvertex):
        """Write the decoder region for the given subvertex."""
        subvertex.spec.comment("# Decoders Region")
        subvertex.spec.switchWriteFocus(self.REGIONS.DECODERS)

        decoders = self.decoders.get_merged_decoders()

        for n in range(subvertex.lo_atom, subvertex.hi_atom + 1):
            # Write the decoders for all the atoms within this subvertex
            for d in range(self.n_output_dimensions):
                subvertex.spec.write(
                    data=parameters.s1615(decoders[n][d] / self.dt)
                )

    def write_region_output_keys(self, subvertex):
        """Write the output keys region for the given subvertex."""
        subvertex.spec.comment("# Output Keys Region")
        subvertex.spec.switchWriteFocus(self.REGIONS.OUTPUT_KEYS)
        for k in subvertex.output_keys:
            subvertex.spec.write(data=k)

    def generate_routing_info(self, subedge):
        """Generate a key and mask for the given subedge."""
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        i = self.decoders.edge_index(subedge.edge)
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

        return key
