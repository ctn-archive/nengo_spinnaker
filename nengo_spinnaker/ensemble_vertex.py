import os
import numpy as np

import nengo
import nengo.builder
from pacman103.lib import graph, data_spec_gen, lib_map, parameters
from pacman103.front.common import enums

from . import decoder_bin


class EnsembleVertex(graph.Vertex):
    """PACMAN Vertex for an Ensemble."""

    REGIONS = enums.enum1(
        'SYSTEM',
        'BIAS',
        'ENCODERS',
        'DECODERS',
        'OUTPUT_KEYS'
    )

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
                    ens.neurons.n_neurons, rng=rng
                )
            if hasattr(ens.intercepts, 'sample'):
                ens.intercepts = ens.intercepts.sample(
                    ens.neurons.n_neurons, rng=rng
                )
            ens.neurons.set_gain_bias(ens.max_rates, ens.intercepts)

        self.bias = ens.neurons.bias
        self.gain = ens.neurons.gain
        self.tau_rc = ens.neurons.tau_rc
        self.tau_ref = ens.neurons.tau_ref

        self.n_input_dimensions = ens.dimensions

        # Set up encoders
        if ens.encoders is None:
            self.encoders = ens.neurons.default_encoders(ens.dimensions, rng)
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

        # Set up decoder bin
        self.decoders = decoder_bin.DecoderBin(rng)

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

    # FOR UPSTREAM CHANGES
    def sdram_usage(self, lo_atom, hi_atom):
        """Return the amount of SDRAM used for the specified atoms."""
        # At the moment this is the same as the DTCM usage, though this may
        # change.
        return self.dtcm_usage(lo_atom, hi_atom)

    # FOR UPSTREAM CHANGES
    def dtcm_usage(self, lo_atom, hi_atom):
        """Return the amount of DTCM used for the specified atoms."""
        n_atoms = hi_atom - lo_atom
        return sum([
            self.sizeof_region_system(),
            self.sizeof_region_bias(n_atoms),
            self.sizeof_region_encoders(n_atoms),
            self.sizeof_region_decoders(n_atoms),
            self.sizeof_region_output_keys(),
        ])

    # FOR UPSTREAM CHANGES
    def cpu_usage(self, lo_atom, hi_atom):
        """Return the CPU utilisation for the specified atoms."""
        # TODO: Calculate this
        return 0

    # FOR UPSTREAM CHANGES
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
        return (
            partition_data_object,
            lib_map.Resources(
                self.cpu_usage(lo_atom, hi_atom),
                self.dtcm_usage(lo_atom, hi_atom),
                self.sdram_usage(lo_atom, hi_atom)
            )
        )

    # TO BE DEPRECATED
    def get_requirements_per_atom(self):
        """Return some indication of the cost of including a single atom on a
        processing core.
        """
        return lib_map.Resources(
            1,
            self.dtcm_usage(0, self.atoms) / self.atoms,
            self.sdram_usage(0, self.atoms) / self.atoms
        )

    def build_decoders(self):
        """Build the decoders for the Ensemble."""
        for e in self.out_edges:
            print e.conn
            e.index = self.decoders.get_decoder_index(e)

    def generateDataSpec(self, processor, subvertex, dao):
        """Generate the data spec for the given subvertex."""
        # Create a spec for the subvertex
        subvertex.spec = data_spec_gen.DataSpec(processor, dao)
        subvertex.spec.initialise(0xABCD, dao)
        subvertex.spec.comment("# Nengo Ensemble")

        # Finalise the values for this Ensemble
        # Encode any constant inputs, and add to the biases
        self.bias = np.dot(self.encoders, self.direct_input)

        # Generate the list of decoders, and the list of ouput keys
        subvertex.output_keys = list()
        x, y, p = processor.get_coordinates()
        for (i, w) in enumerate(self.decoders.decoder_widths):
            # Generate the routing keys for each dimension
            for d in range(e.edge.n_dimensions):
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

        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            os.path.join(dao.get_binaries_directory(), 'nengo_ensemble.aplx'),
            x, y, p
        )

        return (executable_target, list(), list())

    def reserve_regions(self, subvertex):
        """Reserve sufficient space for the regions in the spec."""
        for (region, sizeof) in [
            (self.REGIONS.SYSTEM, self.sizeof_region_system),
            (self.REGIONS.BIAS, self.sizeof_region_bias),
            (self.REGIONS.ENCODERS, self.sizeof_region_encoders),
            (self.REGIONS.DECODERS, self.sizeof_region_decoders),
            (self.REGIONS.OUTPUT_KEYS, self.sizeof_region_output_keys),
        ]:
            subvertex.spec.reserveMemRegion(region, sizeof(subvertex.n_atoms))

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
        # subvertex.spec.write(data=... FILTER DECAY ...)
        # subvertex.spec.write(data=... FILTER DECAY COMPLEMENT ...)

    def write_region_bias(self, subvertex):
        """Write the bias region for the given subvertex."""
        subvertex.spec.switchWriteFocus(self.REGIONS.BIAS)
        for n in range(subvertex.lo_atom, subvertex.hi_atom + 1):
            # Write the bias for all atoms within this subvertex
            subvertex.spec.write(data=parameters.s1615(self.bias[n]))

    def write_region_encoders(self, subvertex):
        """Write the encoder region for the given subvertex."""
        subvertex.spec.switchWriteFocus(self.REGIONS.ENCODERS)
        subvertex.spec.comment("# Encoders Region")
        for n in range(subvertex.lo_atom, subvertex.hi_atom + 1):
            for d in range(self.data.n_input_dimensions):
                subvertex.spec.write(
                    data=parameters.s1615(
                        self.data.encoders[n, d] * self.data.gain[n]
                    )
                )

    def write_region_decoders(self, subvertex):
        """Write the decoder region for the given subvertex."""
        subvertex.spec.comment("# Decoders Region")
        subvertex.spec.switchWriteFocus(self.REGIONS.DECODERS)
        raise NotImplementedError

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
        mask = 0xFFFFFFE0

        return key, mask
