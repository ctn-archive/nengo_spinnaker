from pacman103.lib import graph
import nengo


class NengoEdge(graph.Edge):
    def __init__(self, conn, pre, post, constraints=None, label=None,
                 filter_is_accumulatory=True):
        super(NengoEdge, self).__init__(
            pre, post, constraints=constraints, label=label
        )
        self.index = None  # Used in generating routing keys
        self.conn = conn   # Handy reference
        self._filter_is_accumulatory = filter_is_accumulatory

    @property
    def width(self):
        if isinstance(self.conn, int):
            return self.conn
        elif isinstance(self.conn.post, nengo.Ensemble):
            return self.conn.post.dimensions
        elif isinstance(self.conn.post, nengo.Node):
            return self.conn.post.size_in

    def __getattr__(self, name):
        """Redirect missed attributes to the connection."""
        return getattr(self.conn, name)


class DecoderEdge(NengoEdge):
    """Edge representing a connection from an Ensemble."""
    pass


class InputEdge(NengoEdge):
    """Edge representing a connection from a Node via an ReceiveVertex."""
    pass
