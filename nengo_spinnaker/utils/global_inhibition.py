"""Utilities for global inhibition connections.
"""
import numpy as np

import nengo
import nengo.objects
import nengo.utils.builder

from nengo_spinnaker import edges


def create_inhibition_connection(conn):
    return GlobalInhibitionConnection(conn, add_to_container=False)


def create_inhibition_edge(conn, prevertex, postvertex):
    # Ensure that an inhibitory edge does not already exist for the post vertex
    if postvertex.inhibitory_edge is not None:
        raise NotImplementedError("Only one inhibitory connection may be made "
                                  "to an ensemble.")
    c = create_inhibition_connection(conn)
    e = GlobalInhibitionEdge(conn=c, pre=prevertex, post=postvertex)
    postvertex.inhibitory_edge = e
    return e


def get_not_inhibited_connections(connections):
    for c in connections:
        if not isinstance(c, GlobalInhibitionConnection):
            yield c


class GlobalInhibitionConnection(nengo.Connection):
    """Create a new Connection with the replace the given argument which
    represents a connection to a 1D receiver.
    """
    _skip_check_shapes = True

    def __init__(self, conn):
        # Assert that the connection is an appropriate Connection to modify
        assert(isinstance(conn.pre, nengo.objects.Ensemble))
        assert(isinstance(conn.post, nengo.objects.Neurons))

        # Ensure that the transform is uniform
        ts_ = nengo.utils.builder.full_transform(conn, allow_scalars=False)
        ts_ = ts_.reshape(ts_.size)
        if not np.all(ts_[0] == ts_):
            raise ValueError("Only support global inhibition connections "
                             "transforms are of the form [[k]]*n.")

        # Connect as required
        self._masked_conn = conn
        self._pre = conn.pre
        self._post = GlobalInhibitionTarget(conn.pre.size_out,
                                            add_to_container=False)
        self.function = conn.function
        self.transform = ts_[0]

        self._preslice = conn._preslice
        self._postslice = conn._postslice

        self.synapse = conn.synapse


class GlobalInhibitionTarget(nengo.Node):
    def __init__(self, size_in=1):
        self.size_in = size_in
        self.size_out = 0


class GlobalInhibitionEdge(edges.NengoEdge):
    @property
    def synapse(self):
        return self.conn.synapse
