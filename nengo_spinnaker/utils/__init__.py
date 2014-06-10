import nengo

from . import connections
from . import fixpoint as fp
from . import nodes


def get_connection_width(connection):
    """Return the width of a Connection."""
    if isinstance(connection, int):
        return connection
    elif isinstance(connection.post, nengo.Ensemble):
        return connection.post.dimensions
    elif isinstance(connection.post, nengo.Node):
        return connection.post.size_in
    else:
        raise NotImplementedError
