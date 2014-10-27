"""Utilities for representing synapses/filters.
"""

import collections
import numpy as np

from . import fixpoint as fp
from ..connection import _filter_types, LowpassFilterParameter
from ..spinnaker import regions


def get_filter_from_connection(connection):
    """Return a filter object representing the connection.
    """
    return _filter_types[connection.synapse.__class__].from_synapse(
        connection.width, connection.synapse,
        getattr(connection, 'is_accumulatory', True), connection.modulatory
    )


def get_combined_filters(connections):
    """Return the minimum set of filters required to filter given connections.

    Also returns a mapping of connection to filter index.
    """
    # Create a dictionary of filter to connections
    filter_connections = collections.defaultdict(list)
    for c in connections:
        f = get_filter_from_connection(c)
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
