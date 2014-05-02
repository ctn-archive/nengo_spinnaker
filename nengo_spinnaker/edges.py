from pacman103.lib import graph
import nengo


def get_connection_width(connection):
    """Return the width of a Connection."""
    if isinstance(connection, int):
        return connection
    elif isinstance(connection.post, nengo.Ensemble):
        return connection.post.dimensions
    elif isinstance(connection.post, nengo.Node):
        return connection.post.size_in


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
        get_connection_width(self.conn)

    def __getattr__(self, name):
        """Redirect missed attributes to the connection."""
        return getattr(self.conn, name)


class DecoderEdge(NengoEdge):
    """Edge representing a connection from an Ensemble."""
    pass


class InputEdge(NengoEdge):
    """Edge representing a connection from a Node via an ReceiveVertex."""
    pass
