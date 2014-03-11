from pacman103.lib import graph


class InputEdge(graph.Edge):
    """
    Represents input to an object from the host.  Used to connect an Rx
    element to some other application on a core.

    :param prevertex: Source vertex
    :param postvertex: Destination vertex
    :param constraints: TBD
    :param label: Label for the edge

    .. todo::
        - Neaten this when we refactor
    """

    @property
    def width(self):
        """The number of dimensions represented by this Edge."""
        return self.postvertex.data.D_in

    @property
    def start(self):
        """The offset that represents the zeroth dimension of this edge from
        the view of the Rx component."""
        self_index = self.prevertex.out_edges.index(self)
        edges = self.prevertex.out_edges[0:self_index]
        return sum([e.width for e in edges])
