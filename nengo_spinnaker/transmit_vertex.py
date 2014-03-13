from pacman103.lib import graph
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
