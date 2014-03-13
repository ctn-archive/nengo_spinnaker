"""Bin-packing for Nodes.
"""


class AssignedNode(object):
    """The indexing for an assigned Node."""
    def __init__(self, node, index, width):
        self._node = node
        self._index = index
        self._width = width

    @property
    def node(self):
        return self._node

    @property
    def index(self):
        return self._index

    @property
    def width(self):
        return self._width


class AssignedNodeBin(object):
    """Manages assigning Nodes to slots of available dimensions.
    """
    def __init__(self, max_d, width_f=lambda n: n.size_in):
        """
        :param max_d: Maximum number of dimensions to assign.
        :param width_f: Function which returns the width of a Node.
        """
        self._node_list = list()
        self._max_d = max_d
        self._width_f = width_f

    @property
    def n_assigned_dimensions(self):
        """Get the number of dimensions which have been assigned."""
        return sum(map(lambda n: n.width, self._node_list))

    @property
    def remaining_space(self):
        return self._max_d - self.n_assigned_dimensions

    @property
    def nodes(self):
        """Return the Nodes contained in this bin."""
        for an in self._node_list:
            yield an.node

    def append(self, node):
        """Add a Node to the bin."""
        if not self._width_f(node) <= self.remaining_space:
            raise ValueError(
                "Cannot add node with %d dimensions, only %d remaining."
                % (self._width_f(node), self.remaining_space)
            )

        self._node_list.append(
            AssignedNode(node, self.n_assigned_dimensions, self._width_f(node))
        )
