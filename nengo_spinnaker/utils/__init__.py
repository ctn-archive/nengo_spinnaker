import nengo

from . import builder
from . import connections
from . import decoders
from . import fixpoint as fp
from . import keyspaces
from . import nodes
from . import probes
from . import vertices


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
