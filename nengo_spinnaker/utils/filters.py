"""Utilities for representing synapses/filters.
"""

import numpy as np
from six import iteritems

from . import fixpoint as fp
from ..connections.reduced import LowpassFilterParameter
from ..spinnaker import regions


def get_filter_regions(filter_keyspaces, dt, width):
    """Return the filter and routing region for the given map of filters to
    keyspaces.

    Returns a tuple of (filter region, routing region).
    """
    # Build a list of filters and a map of keyspaces to filter IDs
    filters = list()
    filter_ids = dict()

    for i, (f, keyspaces) in enumerate(iteritems(filter_keyspaces)):
        # Store the filter
        filters.append(f)

        # Add the keyspace to ID map
        for ks in keyspaces:
            filter_ids[ks] = i

    # Create and return the appropriate regions
    return (make_filter_region(filters, dt, width),
            make_routing_region(filter_ids))


def make_filter_region(filters, dt, width):
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
    widths = np.array([width] * len(filters), dtype=np.uint32)

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
    return regions.KeysRegion(
        filter_ids.keys(),
        extra_fields=[
            lambda ks, i: ks.filter_mask,  # Filter mask
            lambda ks, i: filter_ids[ks],  # Filter index
            lambda ks, i: ks.mask_d,       # Dimension mask
        ],
        prepend_n_keys=True)
