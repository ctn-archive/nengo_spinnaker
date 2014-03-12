import os
import numpy as np

from pacman103.lib import graph, data_spec_gen, lib_map, parameters
from pacman103.front.common import enums


class EnsembleVertex(graph.Vertex):
    """PACMAN Vertex for an Ensemble."""

    REGIONS = enums.enum1(
        'SYSTEM',
        'BIAS',
        'ENCODERS',
        'DECODERS',
        'OUTPUT_KEYS'
    )

    def __init__(self, ens, constraints=None):
        """Create a new EnsembleVertex using the given Ensemble to generate
        appropriate parameters.

        :param ens: Ensemble to represent in the simulation.
        """
        # Save a reference to the ensemble before unpacking some useful values
        self._ens = ens

        # For constant value injection
        self.direct_input = np.zeros(self._ens.dimensions)

        # Create the vertex
        super(EnsembleVertex, self).__init__(
            self._ens.n_neurons, constraints=constraints, label=ens.label
        )

    @property
    def _tau_ref_in_steps(self):
        # TODO: Check this
        return self._ens.neurons.tau_ref * 10**-3

    @property
    def _one_over_tau_rc(self):
        return 1. / self._encs.neurons.tau_rc

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

    def generateDataSpec(self, processor, subvertex, dao):
        """Generate the data spec for the given subvertex."""
        # Create a spec for the subvertex
        subvertex.spec = data_spec_gen.DataSpec(processor, dao)
        subvertex.spec.initialise(0xABCD, dao)
        subvertex.spec.comment("# Nengo Ensemble")

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
        subvertex.spec.comment("""# System Region
        # -------------
        # 1. Number of input dimensions
        # 2. Number of output dimensions
        # 3. Number of neurons
        # 4. Machine time step in us
        # 5. tau_ref in number of steps
        # 6. tau_rc ^ -1
        # 7. Filter decay (TO BE CHANGED)
        # 8. 1 - Filter decay (TO BE CHANGED)
        """)
        subvertex.spec.write(data=self.n_input_dimensions)
        subvertex.spec.write(data=self.n_output_dimensions)
        subvertex.spec.write(data=subvertex.n_atoms)
        subvertex.spec.write(data=int(0.001 * 10**-6))
        subvertex.spec.write(data=self._tau_ref_in_steps)
        subvertex.spec.write(data=parameters.s1615(self._one_over_tau_rc))
        # subvertex.spec.write(data=... FILTER DECAY ...)
        # subvertex.spec.write(data=... FILTER DECAY COMPLEMENT ...)

    def write_region_bias(self, subvertex):
        """Write the bias region for the given subvertex."""
        raise NotImplementedError

    def write_region_encoders(self, subvertex):
        """Write the encoder region for the given subvertex."""
        raise NotImplementedError

    def write_region_decoders(self, subvertex):
        """Write the decoder region for the given subvertex."""
        raise NotImplementedError

    def write_region_output_keys(self, subvertex):
        """Write the output keys region for the given subvertex."""
        raise NotImplementedError
