import os

from pacman103.lib import data_spec_gen, graph, lib_map
from pacman103.front.common import enums
from . import node_bin


class TransmitVertex(graph.Vertex):
    """PACMAN Vertex for an object which receives input from Nodes and
    transmits it to the host.
    """

    REGIONS = enums.enum1(
        'SYSTEM'
    )
    MAX_DIMENSIONS = 64

    def __init__(self, time_step=1000, constraints=None, label=None):
        # Dimension management
        self._assigned_dimensions = 0
        self._assigned_nodes = node_bin.AssignedNodeBin(
            self.MAX_DIMENSIONS, lambda n: n.size_in
        )

        # Create the vertex
        super(TransmitVertex, self).__init__(
            1, constraints=constraints, label=label
        )

    @property
    def remaining_dimensions(self):
        return self._assigned_nodes.remaining_space

    def assign_node(self, node):
        """Assign a Nengo Node to this TransmitVertex."""
        self._assigned_nodes.append(node)

    @property
    def nodes(self):
        """Return the Nodes assigned to this TransmitVertex."""
        return self._assigned_nodes.nodes

    def generateDataSpec(self, processor, subvertex, dao):
        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            os.path.join(dao.get_binaries_directory(), 'nengo_tx.aplx'),
            x, y, p
        )

        # Generate the spec
        spec = data_spec_gen.DataSpec(processor, dao)
        spec.initialise(0xABCE, dao)
        spec.comment("# Nengo Tx Component")

        spec.endSpec()
        spec.closeSpecFile()

        return (executable_target, list(), list())

    def generate_routing_info(self, subedge):
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        key = (x << 24) | (y << 16) | ((p-1) << 11)
        mask = 0xFFFFFFE0

        return key, mask
