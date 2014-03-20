"""Object collections for the Nengo/SpiNNaker Integration
"""

import numpy as np


class DecoderBinEntry(object):
    def __init__(self, decoder, func, transform=None):
        self.decoder = decoder
        self.func = func
        self.trans = transform


class DecoderBin(object):
    """A bin for decoders."""
    def __init__(self, rng):
        self._decoders = list()
        self._decoders_by_func= dict()
        self._decoders_edges = dict()
        self.rng = rng

    @property
    def decoder_widths(self):
        """Get the width of each decoder."""
        return [decoder.decoder.shape[1] for decoder in self._decoders]

    @property
    def width(self):
        return sum(self.decoder_widths)

    def get_decoder_index(self, e):
        """Get the decoder index for a given edge.

        ..todo::
          - Check function equivalency by comparing sampled points?
          - Check transform equivalency by comparing matrix values rather
              than entire matrix (e.g., dimension reduction can use the same
              transform despite being a different matrix.)
        """
        # Check if the decoder already exists for this function and transform
        for (i,dec) in enumerate(self._decoders):
            if dec.func == e.function and dec.trans == conn.transform:
                self._decoders_edges[e] = i
                return i

        # Check if the decoder already exists for this function
        decoder = self._decoders_by_func.get(e.function, None)
        if decoder is None:
            # Compute the decoder
            eval_points = e.eval_points
            if eval_points is None:
                eval_points = e.prevertex.eval_points
            activities = e.pre.activities(eval_points)

            if e.function is None:
                targets = eval_points
            else:
                targets = np.array(
                    [e.function(ep) for ep in eval_points]
                )
                if targets.ndim < 2:
                    targets.shape = targets.shape[0], 1
            decoder = e.decoder_solver(activities, targets, self.rng)
            self._decoders_by_func[e.function] = decoder

        # Combine the decoder with the transform and record it in the list
        decoder = np.dot(decoder, np.asarray(e.transform).T)
        self._decoders.append(
            DecoderBinEntry(decoder, e.transform, e.function)
        )
        i = len(self._decoders) - 1
        self._decoders_edges[e] = i

        return i

    def get_merged_decoders(self):
        """Get a merged decoder for the bin."""
        return np.hstack([d.decoder for d in self._decoders])

    def edge_index(self, edge):
        """Get the index of the decoder which this edge utilises."""
        return self._decoders_edges[edge]


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


class FilterBinEntry(object):
    def __init__(self, filter_value, edges=[]):
        self._value = filter_value
        self._edges = edges

    def add_edge(self, edge):
        self._edges.append(edge)

    def get_filter_tc(self, dt):
        """Get the filter time constant and complement for the given dt."""
        tc = np.exp(-dt/self._value)
        return (tc, 1. - tc)

    def get_keys_masks(self, subvertex):
        """Return the set of keys for edges which use this filter arriving at
        the given subvertex."""
        # Get the list of subedges
        subedges = []
        for edge in self._edges:
            for subedge in edge.subedges:
                if subedge.postsubvertex == subvertex:
                    subedges.append(subedge)

        # Get the list of keys
        kms = []
        for subedge in subedges:
            km = subedge.presubvertex.vertex.get_routing_info(subedge)
            kms.append(km)

        return kms


class FilterCollection(object):
    """A collection of filters."""
    def __init__(self):
        self._entries = {}

    def add_edge(self, edge):
        """Add the given edge to the filter collection."""
        # Create a new filter if necessary
        if not edge.filter in self._entries.keys():
            self._entries[edge.filter] = FilterBinEntry(edge.filter)
        self._entries[edge.filter].add_edge(edge)

    @property
    def filter_values(self):
        """Return a list of filter values."""
        return self._entries.keys()

    def filter_tcs(self, dt):
        """Return a list of tupled filter time constants and complements."""
        return [f.get_filter_tc(dt) for f in self._entries]

    def get_indexed_keys_masks(self, subvertex):
        """Return a list of keys and masks for each filter in the collection
        for the given postsubvertex.
        """
        return [f.get_keys_masks(subvertex) for f in self._entries]
