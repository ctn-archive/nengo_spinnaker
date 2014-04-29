import os

from pacman103.lib import data_spec_gen, graph, lib_map
from pacman103.front.common import enums
from .. import collections


class ReceiveVertex(graph.Vertex):
    """PACMAN Vertex for an object which receives input from Nodes on the host
    and forwards it to connected Ensembles.
    """

    REGIONS = enums.enum1(
        'SYSTEM',
        'OUTPUT_KEYS'
    )
    MAX_DIMENSIONS = 64

    model_name = "nengo_rx"

    def __init__(self, time_step=1000, constraints=None, label=None):
        # Dimension management
        self._assigned_dimensions = 0
        self._assigned_nodes = collections.AssignedNodeBin(
            self.MAX_DIMENSIONS, lambda n: n.size_out
        )

        # Create the vertex
        super(ReceiveVertex, self).__init__(
            1, constraints=constraints, label=label
        )

    def get_maximum_atoms_per_core(self):
        return 1

    def get_resources_for_atoms(self, lo_atom, hi_atom, n_machine_time_steps,
                                machine_time_step_us, partition_data_object):
        return lib_map.Resources(1, 1, 1)

    @property
    def n_assigned_dimensions(self):
        return self._assigned_nodes.n_assigned_dimensions

    @property
    def remaining_dimensions(self):
        return self._assigned_nodes.remaining_space

    def assign_node(self, node):
        """Assign a Nengo Node to this ReceiveVertex."""
        self._assigned_nodes.append(node)

    def node_index(self, node):
        """Get the offset of this Node in the ReceiveVertex."""
        return self._assigned_nodes.node_index(node)

    @property
    def nodes(self):
        """Return the Nodes assigned to this ReceiveVertex."""
        return self._assigned_nodes.nodes

    def sizeof_region_system(self):
        """Get the size (in bytes) of the SYSTEM region."""
        # 2 words
        return 4 * 2

    def sizeof_region_output_keys(self):
        """Get the size (in bytes) of the OUTPUT_KEYS region."""
        # 1 word per edge
        return 4 * len(self.out_edges)

    def generateDataSpec(self, processor, subvertex, dao):
        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            os.path.join(dao.get_common_binaries_directory(), 'nengo_rx.aplx'),
            x, y, p
        )

        # Generate the spec
        subvertex.spec = data_spec_gen.DataSpec(processor, dao)
        subvertex.spec.initialise(0xABCE, dao)
        subvertex.spec.comment("# Nengo Rx Component")

        # Fill in the spec
        self.reserve_regions(subvertex)
        self.write_region_system(subvertex)
        self.write_region_output_keys(subvertex)

        subvertex.spec.endSpec()
        subvertex.spec.closeSpecFile()

        return (executable_target, list(), list())

    def reserve_regions(self, subvertex):
        subvertex.spec.reserveMemRegion(
            self.REGIONS.SYSTEM,
            self.sizeof_region_system()
        )
        subvertex.spec.reserveMemRegion(
            self.REGIONS.OUTPUT_KEYS,
            self.sizeof_region_output_keys()
        )

    def write_region_system(self, subvertex):
        subvertex.spec.switchWriteFocus(self.REGIONS.SYSTEM)
        subvertex.spec.comment("""# System Region
        # -------------
        # 1. Number of us between transmitting MC packets
        # 2. Number of dimensions
        """)
        subvertex.spec.write(data=1000/self.n_assigned_dimensions)
        subvertex.spec.write(data=self.n_assigned_dimensions)

    def write_region_output_keys(self, subvertex):
        subvertex.spec.switchWriteFocus(self.REGIONS.OUTPUT_KEYS)
        subvertex.spec.comment("# Output Keys")

        keys = [self.generate_routing_info(subedge)[0] for subedge in
                subvertex.out_subedges]

        for (base, subedge) in zip(keys, subvertex.out_subedges):
            for d in range(subedge.edge.width):
                subvertex.spec.write(data=base | d)

    def generate_routing_info(self, subedge):
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        i = self.node_index(subedge.edge.pre)
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

        return key, 0xFFFFFFE0
