from pacman103.lib import data_spec_gen, graph, lib_map
from pacman103.front.common import enums

from ..utils import bins
from .. import vertices


class TransmitVertex(graph.Vertex, vertices.VertexWithFilters):
    """PACMAN Vertex for an object which receives input for a Node and
    transmits it to the host.
    """

    REGIONS = enums.enum1(
        'SYSTEM',
        'FILTERS',
        'FILTER_ROUTING'
    )
    model_name = "nengo_tx"

    def __init__(self, node, dt=0.001, output_period=100, time_step=1000,
                 constraints=None, label=None):
        self.node = node

        self.dt = dt
        self.time_step = time_step
        self.output_period = output_period

        self.filters = bins.FilterCollection()

        # Create the vertex
        super(TransmitVertex, self).__init__(
            1, constraints=constraints, label=label
        )

    def get_maximum_atoms_per_core(self):
        return 1

    def get_resources_for_atoms(self, lo_atom, hi_atom, n_machine_time_steps,
                                machine_time_step_us, partition_data_object):
        return lib_map.Resources(1, 1, 1)

    def sizeof_region_system(self):
        """Get the size (in bytes) of the SYSTEM region."""
        # 5 words
        return 4 * 5

    def generateDataSpec(self, processor, subvertex, dao):
        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            vertices.resource_filename("nengo_spinnaker",
                                       "binaries/%s.aplx" % self.model_name),
            x, y, p
        )

        # Generate the spec
        subvertex.spec = data_spec_gen.DataSpec(processor, dao)
        subvertex.spec.initialise(0xABCE, dao)
        subvertex.spec.comment("# Nengo Tx Component")

        # Fill in the spec
        self.reserve_regions(subvertex)
        self.write_region_system(subvertex)

        if len(self.filters) > 0:
            self.write_region_filters(subvertex)
            self.write_region_filter_keys(subvertex)

        # Close the spec
        subvertex.spec.endSpec()
        subvertex.spec.closeSpecFile()

        return (executable_target, list(), list())

    def reserve_regions(self, subvertex):
        """Reserve sufficient space for the regions in the spec."""
        # TODO Modify the following functions to use write_array rather than
        #  lots of writes.
        subvertex.spec.reserveMemRegion(
            self.REGIONS.SYSTEM,
            self.sizeof_region_system()
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
        subvertex.spec.write(data=self.node.size_in)
        subvertex.spec.write(data=self.time_step)
        subvertex.spec.write(data=self.output_period)

        if len(self.in_edges) > 0:
            subvertex.spec.write(data=len(self.filters))
            subvertex.spec.write(data=self.filters.num_keys(subvertex))
        else:
            subvertex.spec.write(data=0, repeats=2)
