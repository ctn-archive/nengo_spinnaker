"""Utilities for representing synapses/filters.
"""

import collections
import nengo
import numpy as np

from . import fixpoint as fp
from ..spinnaker import regions


class FilterParameter(object):
    """Base class for filter types."""
    def __init__(self, width, is_accumulatory=True, is_modulatory=False):
        # TODO: Is width actually required?  Keep it to maintain compatibility
        #       current C code, but investigate removing it if we can.
        self.width = width
        self.is_accumulatory = is_accumulatory
        self.is_modulatory = is_modulatory

    @classmethod
    def from_synapse(cls, synapse, is_accumulatory=True, is_modulatory=False):
        raise NotImplementedError

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.is_accumulatory == other.is_accumulatory and
                self.is_modulatory == other.is_modulatory and
                self.width == other.width)

    def __hash__(self):
        return hash((hash(self.__class__), hash(self.is_accumulatory),
                     hash(self.is_modulatory), hash(self.width)))


class LinearFilterFilterParameter(FilterParameter):
    def __init__(self, width, num, den, is_accumulatory=True,
                 is_modulatory=False):
        super(LinearFilterFilterParameter, self).__init__(
            width, is_accumulatory, is_modulatory)
        self.num = num
        self.den = den

    @classmethod
    def from_synapse(cls, width, synapse, is_accumulatory=True,
                     is_modulatory=False):
        return cls(width, synapse.num, synapse.den, is_accumulatory,
                   is_modulatory)

    def __eq__(self, other):
        return (super(LinearFilterFilterParameter, self).__eq__(other) and
                self.num == other.num and
                self.den == other.den)

    def __hash__(self):
        return hash((super(LinearFilterFilterParameter, self).__hash__(),
                     hash(self.num), hash(self.den)))


class LowpassFilterParameter(FilterParameter):
    def __init__(self, width, tau, is_accumulatory=True, is_modulatory=False):
        super(LowpassFilterParameter, self).__init__(width, is_accumulatory,
                                                     is_modulatory)
        self.tau = tau if tau is not None else 0.

    @classmethod
    def from_synapse(cls, width, synapse, is_accumulatory=True,
                     is_modulatory=False):
        return cls(width, synapse.tau, is_accumulatory, is_modulatory)

    def __eq__(self, other):
        return (super(LowpassFilterParameter, self).__eq__(other) and
                self.tau == other.tau)

    def __hash__(self):
        return hash((super(LowpassFilterParameter, self).__hash__(),
                     hash(self.tau)))


class AlphaFilterParameter(LowpassFilterParameter):
    # Obviously this is a different type, but it has the same parameters as the
    # lowpass and so can derive from it safely (DRY).
    pass


_FilterTypes = {nengo.synapses.LinearFilter: LinearFilterFilterParameter,
                nengo.Lowpass: LowpassFilterParameter,
                nengo.synapses.Alpha: AlphaFilterParameter, }


def get_combined_filters(connections):
    """Return the minimum set of filters required to filter given connections.

    Also returns a mapping of connection to filter index.
    """
    # Create a dictionary of filter to connections
    filter_connections = collections.defaultdict(list)
    for c in connections:
        f = _FilterTypes[c.synapse.__class__].from_synapse(
            c.width,
            c.synapse,
            getattr(c, 'is_accumulatory', True),
            c.modulatory
        )
        filter_connections[f].append(c)

    # Create a list of filters and a mapping of connection to filter index.
    filters = list()
    filter_indices = dict()

    for (f, cs) in filter_connections.iteritems():
        filters.append(f)

        for c in cs:
            filter_indices[c] = filters.index(f)

    return filters, filter_indices


def get_keyspace_to_filter_map(filter_indices):
    """Convert a mapping from connection to filter index to keyspace to indices
    """
    return {c.keyspace: v for (c, v) in filter_indices.items()}


def get_filter_regions(connections, dt):
    """Return the filter and routing region for the given connections.

    Returns a tuple of (filter region, routing region).
    """
    # Get the minimum set of filters
    filters, filter_ids = get_combined_filters(connections)

    # Create and return the appropriate regions
    return make_filter_region(filters, dt), make_routing_region(filter_ids)


def make_filter_region(filters, dt):
    """Return a region representing parameters for incoming value filters.

    TODO: Support more than just 1st order low-pass filters.
    """
    for f in filters:
        assert isinstance(f, LowpassFilterParameter)

    # Build up the filter coefficients
    cs = np.exp(-dt / np.array([f.tau for f in filters]))
    ics = 1. - cs

    fpc = np.array(fp.bitsk(cs), dtype=np.uint32)
    fpd = np.array(fp.bitsk(ics), dtype=np.uint32)

    # Get the accumulator masks for the filters and the widths of the filters
    masks = np.array([0x0 if f.is_accumulatory else 0xffffffff for f in
                      filters], dtype=np.uint32)
    widths = np.array([f.width for f in filters], dtype=np.uint32)

    # Stack these columns to make the full filter matrix
    f_matrix = np.vstack([fpc, fpd, masks, widths]).T
    assert f_matrix.shape == (len(filters), 4)

    # Once we've computed the data to store, this region is just an
    # unpartitioned matrix region which prepends the number of rows
    # to the block of data when writing out.
    return regions.MatrixRegion(
        matrix=f_matrix,
        prepends=[regions.MatrixRegionPrepends.N_ROWS]
    )


def make_routing_region(filter_ids):
    """Return a region representing filter routing information.

    The data written out is an array, each entry contains:
     - The filter routing key to match
     - The filter mask to apply to incoming packets
     - The index of the filter packets matching this entry should use
     - The mask to apply to get the component index from the key
    """
    filter_ids = get_keyspace_to_filter_map(filter_ids)
    return regions.KeysRegion(
        filter_ids.keys(),
        extra_fields=[
            lambda ks, i: ks.filter_mask,  # Filter mask
            lambda ks, i: filter_ids[ks],  # Filter index
            lambda ks, i: ks.mask_d,       # Dimension mask
        ],
        prepend_n_keys=True)
