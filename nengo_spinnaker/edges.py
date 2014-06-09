import collections
import numpy as np

from pacman103.lib import graph

from . import utils


class DummyConnection(object):
    """Dummy Connection object used in Edges which do not represent a Nengo
    connection"""
    _preslice = None
    _postslice = None

    def __init__(self, pre=None, post=None, transform=1., function=None,
                 solver=None, eval_points=None, synapse=None, 
                 size_in=1, size_out=1):
        self.pre = pre
        self.post = post
        self.function = function
        self.solver = solver
        self.eval_points = eval_points
        self._size_in = size_in
        self._size_out = size_out
        self.synapse = synapse

        if np.array(transform).ndim == 0:
            if size_in != size_out:
                raise NotImplementedError
            self.transform = np.eye(size_out) * transform
        else:
            self.transform = transform

    def _required_transform_shape(self):
        return self._size_out, self._size_in


class Edge(object):
    mask = 0xFFFFFFC0  # Routing mask for this type of edge
    dimension_mask = 0x3F  # Mask to extract the dimension from a key

    def generate_key(self, x, y, p, i):
        """Return a key for the edge, this will be used in conjunction with
        the mask from this class for routing.
        """
        return (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)


class NengoEdge(graph.Edge, Edge):
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
        utils.get_connection_width(self.conn)

    def __getattr__(self, name):
        """Redirect missed attributes to the connection."""
        return getattr(self.conn, name)


class DecoderEdge(NengoEdge):
    """Edge representing a connection from an Ensemble."""
    pass


class InputEdge(NengoEdge):
    """Edge representing a connection from a Node via an ReceiveVertex."""
    pass


class ValueProbeEdge(NengoEdge):
    def __init__(self, probe, pre, post, size_in, size_out, constraints=None,
                 label=None, filter_is_accumulatory=True):
        # Construct a dummy connection, pass to Nengo edge
        conn = DummyConnection(pre=pre._ens, size_in=size_in,
                               size_out=size_out,
                               synapse=probe.conn_args.get('synapse', None))

        super(ValueProbeEdge, self).__init__(
            conn, pre, post, constraints=constraints, label=label
        )
        self.index = None  # Used in generating routing keys
        self.probe = probe
        self._filter_is_accumulatory = filter_is_accumulatory

    @property
    def width(self):
        return self.probe.size_in
