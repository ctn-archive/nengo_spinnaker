import nengo

from . import bins
from . import fixpoint as fp


def totuple(a):
    """Convert any object (e.g., numpy array) to a Tuple.

    http://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
    """
    try:
        return totuple(totuple(i) for i in a)
    except TypeError:
        return a


def get_connection_width(connection):
    """Return the width of a Connection."""
    if isinstance(connection, int):
        return connection
    elif isinstance(connection.post, nengo.Ensemble):
        return connection.post.dimensions
    elif isinstance(connection.post, nengo.Node):
        return connection.post.size_in
