import mock
import nengo
import numpy as np

from ...connections.reduced import (
    LowpassFilterParameter, _filter_types, StandardInputPort
)
from ...connections.intermediate import IntermediateConnection
from ...connections.connection_tree import ConnectionTree
from .. import filters as filter_utils
from ..fixpoint import bitsk


class AlphaFilterParameter(LowpassFilterParameter):
    pass


_filter_types[nengo.synapses.Alpha] = AlphaFilterParameter


def test_get_filter_regions():
    """Test that an appropriate filter and keys region are produced.
    """
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        a.eval_points = np.random.uniform(size=(100, 1))
        b = nengo.Ensemble(100, 1)

        # Create a set of connections.
        # 0 + 1 share a filter
        # 2 is a unique filter
        # 3 is a unique filter
        # 4 + 5 share a filter
        # 6 is a unique filter (because it is accumulatory)
        cs = [
            nengo.Connection(a, b, synapse=0.01),
            nengo.Connection(a, b, transform=0.1, synapse=0.01),
            nengo.Connection(a, b, synapse=0.05),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.05)),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.01)),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.01)),
            nengo.Connection(a, b, synapse=0.01),
        ]

        conns = [IntermediateConnection.from_connection(c) for c in
                 cs]
        conns[6].is_accumulatory = False

    # Convert each connection to an intermediate connection with a unique
    # "keyspace".
    for c in conns:
        c.keyspace = mock.Mock()

    # Assign a key and mask to each keyspace
    for i, c in enumerate(conns):
        c.keyspace.key.return_value = i
        c.keyspace.filter_mask.return_value = 0xffffffff

    # Create the connection tree
    tree = ConnectionTree.from_intermediate_connections(conns)

    # Get all the inputs for B, Standard Input Port, build the filter regions
    # for those.
    inputs = tree.get_incoming_connections(b)[StandardInputPort]

    # Get the filter regions
    dt = 0.001
    filter_region, filter_routing_region = \
        filter_utils.get_filter_regions(inputs, dt, 1)

    # Assert the size of these regions is appropriate: Filter region should be
    # 4 * n_filters (4 words per filter) + 1 (n filters).
    assert filter_region.sizeof(slice(0, 100)) == len(inputs) * 4 + 1

    # Keys region should be 4 * n_connections (4 words per connection) + 1
    assert filter_routing_region.sizeof(slice(0, 100)) == len(conns) * 4 + 1


def test_filter_region():
    """Test that the region writes out filter parameters correctly.
    """
    # Create 2 filter entries
    filters = [
        LowpassFilterParameter(0.05, True),
        LowpassFilterParameter(0.01, False),
    ]

    # Create the filter region
    dt = 0.01
    width = 2
    filter_region = filter_utils.make_filter_region(filters, dt, width=width)

    # Check the size is correct
    assert filter_region.sizeof(slice(0, 1000)) == 2*4 + 1

    # Check the data is correct
    sr = filter_region.create_subregion(slice(25, 50), 0)
    data = np.frombuffer(sr.data, dtype=np.uint32)

    assert len(data) == 2*4 + 1
    assert data[0] == 2  # Number of filters
    for i, f in enumerate(filters):
        assert data[1 + 4*i] == bitsk(np.exp(-dt / f.tau))
        assert data[2 + 4*i] == bitsk(1. - np.exp(-dt / f.tau))
        assert data[3 + 4*i] == 0x0 if f.is_accumulatory else 0xffffffff
        assert data[4 + 4*i] == width


def test_filter_routing_region():
    """Test that the region writes out filter routing values correctly.
    """
    # Create a mock keyspace to check that values are collected correctly.
    keyspace = mock.Mock(spec_set=['key', 'filter_mask', 'mask_d'])
    keyspace.key.return_value = 0xCAFECAFE
    keyspace.filter_mask = 0xBEEFBEEF
    keyspace.mask_d = 0xDEADABCD

    # Create a mapping from keyspace to filter ID
    filter_ids = {keyspace: 0xFEFEFFFF}

    # Create the region
    routing_region = filter_utils.make_routing_region(filter_ids)
    assert routing_region.sizeof(slice(0, 100)) == 4+1

    # Create a subregion version and ensure that the data stored within is
    # correct.
    sr = routing_region.create_subregion(slice(0, 100), 0)
    data = np.frombuffer(sr.data, dtype=np.uint32)
    assert data[0] == 1
    assert data[1:].tolist() == [keyspace.key(), keyspace.filter_mask,
                                 filter_ids[keyspace], keyspace.mask_d]
