from pacman103.lib import graph, data_spec_gen, lib_map
from pacman103.front.common import enums

from .utils import bins
from . import vertices


class FilterVertex(graph.Vertex, vertices.VertexWithFilters):
    """PACMAN Vertex for a filtered external output node."""

    REGIONS = enums.enum1(
        'SYSTEM',
        'OUTPUT_KEYS',
        'FILTERS',
        'FILTER_ROUTING',
    )

    model_name = "nengo_filter"

    def __init__(self, dimensions, output_id, dt=0.001, time_step=1000,
                 output_period=100, constraints=None, label='filter'):
        """Create a new FilterVertex

        :param dimensions: number of values
        :param output_id: id key to place in packet routing
        :param time_step: Machine timestep (in microseconds)
        :param output_period: Time between output events (in ticks)
        """
        self.time_step = time_step
        self.output_id = output_id
        self.output_period = output_period
        self.dt = dt

        self.dimensions = dimensions

        self.filters = bins.FilterCollection()

        # Create the vertex
        super(FilterVertex, self).__init__(1,
                                           constraints=constraints,
                                           label=label)

    def sizeof_region_system(self):
        """Get the size (in bytes) of the SYSTEM region."""
        # 5 words, 4 bytes per word
        return 5 * 4

    def sizeof_region_output_keys(self):
        """Get the size (in bytes) of the OUTPUT_KEYS region."""
        # 1 word per output dimension
        return 4 * self.dimensions

    def sdram_usage(self, lo_atom, hi_atom):
        """Return the amount of SDRAM used for the specified atoms."""
        # At the moment this is the same as the DTCM usage, though this may
        # change.
        return self.dtcm_usage(lo_atom, hi_atom)

    def dtcm_usage(self, lo_atom, hi_atom):
        """Return the amount of DTCM used for the specified atoms."""
        return sum([
            self.sizeof_region_system(),
            self.sizeof_region_output_keys(),
            self.sizeof_region_filters(),
            5 * 3 * 4 * len(self.filters)  # Assume that we will have at most 5
                                           # subvertices feeding into a given
                                           # filter. TODO Improve when possible
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
        return 1

    def generateDataSpec(self, processor, subvertex, dao):
        """Generate the data spec for the given subvertex."""
        # Create a spec for the subvertex
        subvertex.spec = data_spec_gen.DataSpec(processor, dao)
        subvertex.spec.initialise(0xABCE, dao)
        subvertex.spec.comment("# Nengo Output Filter")

        subvertex.output_keys = list()
        x, y, p = processor.get_coordinates()

        for d in range(self.dimensions):
            subvertex.output_keys.append((x << 24) | (y << 16) |
                                         ((p-1) << 11) |
                                         (self.output_id << 6) | d)

        # Fill in the spec
        self.reserve_regions(subvertex)
        self.write_region_system(subvertex)
        self.write_region_output_keys(subvertex)

        if len(self.filters) > 0:
            self.write_region_filters(subvertex)
            self.write_region_filter_keys(subvertex)

        # Close the spec
        subvertex.spec.endSpec()
        subvertex.spec.closeSpecFile()

        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            vertices.resource_filename("nengo_spinnaker",
                                       "binaries/%s.aplx" % self.model_name),
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
            self.REGIONS.OUTPUT_KEYS,
            self.sizeof_region_output_keys()
        )
        if len(self.filters) > 0:
            subvertex.spec.reserveMemRegion(
                self.REGIONS.FILTERS,
                self.sizeof_region_filters()
            )
            subvertex.spec.reserveMemRegion(
                self.REGIONS.FILTER_ROUTING,
                self.sizeof_region_filter_keys(subvertex)
            )

    def write_region_system(self, subvertex):
        """Write the system region for the given subvertex."""
        subvertex.spec.switchWriteFocus(self.REGIONS.SYSTEM)
        subvertex.spec.comment("""# System Region
        # -------------
        # 1. Number of dimensions
        # 2. Machine time step in us
        # 3. Output period in ticks
        # 4. Number of filters
        # 5. Number of filter keys
        """)
        subvertex.spec.write(data=self.dimensions)
        subvertex.spec.write(data=self.time_step)
        subvertex.spec.write(data=self.output_period)

        if len(self.in_edges) > 0:
            subvertex.spec.write(data=len(self.filters))
            subvertex.spec.write(data=self.filters.num_keys(subvertex))
        else:
            subvertex.spec.write(data=0, repeats=2)

    def write_region_output_keys(self, subvertex):
        """Write the output keys region for the given subvertex."""
        subvertex.spec.comment("# Output Keys Region")
        subvertex.spec.switchWriteFocus(self.REGIONS.OUTPUT_KEYS)
        for k in subvertex.output_keys:
            subvertex.spec.write(data=k)

    def generate_routing_info(self, subedge):
        """Generate a key and mask for the given subedge."""
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        i = self.output_id
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

        return key, 0xFFFFFFE0
