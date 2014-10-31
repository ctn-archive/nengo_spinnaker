import mock
import nengo
import numpy as np
import pytest

from ...connections.reduced import LowpassFilterParameter, _filter_types
from ...connections.intermediate import IntermediateConnection
from .. import filters as filter_utils
from ..fixpoint import bitsk


class AlphaFilterParameter(LowpassFilterParameter):
    pass


_filter_types[nengo.synapses.Alpha] = AlphaFilterParameter


@pytest.fixture(scope='function')
def sample_network():
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)

        # Create a set of connections.
        # 0 + 1 share a filter
        # 2 is a unique filter
        # 3 is a unique filter
        # 4 + 5 share a filter
        # 6 is a unique filter (because it is accumulatory)
        # 7 is a unique filter (because it is modulatory)
        cs = [
            nengo.Connection(a, b, synapse=0.01),
            nengo.Connection(a, b, transform=0.1, synapse=0.01),
            nengo.Connection(a, b, synapse=0.05),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.05)),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.01)),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.01)),
            nengo.Connection(a, b, synapse=0.01),
            nengo.Connection(a, b, synapse=0.05, modulatory=True),
        ]

        conns = [IntermediateConnection.from_connection(c) for c in
                 cs]
        conns[6].is_accumulatory = False

    return model, conns


def test_get_combined_filters(sample_network):
    """Test that filters may be combined correctly.
    """
    (_, conns) = sample_network

    # Construct the minimum set of filters and provide indices from connections
    # into the set of filters.
    filters, filter_indices = filter_utils.get_combined_filters(conns)

    # Assert that there are the correct number of filters and that the
    # connection indices are appropriate.
    assert len(filters) == 6

    for n in [2, 3, 6, 7]:  # Not shared filters
        assert filter_indices[conns[n]] not in [
            i for (c, i) in filter_indices.items() if c is not conns[n]]

    # Shared filters
    assert filter_indices[conns[0]] == filter_indices[conns[1]]
    assert filter_indices[conns[4]] == filter_indices[conns[5]]

    # Assert that parameters have been grabbed correctly
    for i in [0, 2, 3, 4, 7]:
        assert filters[filter_indices[conns[i]]].tau == conns[i].synapse.tau
        assert filters[filter_indices[conns[i]]].is_accumulatory

    assert filters[filter_indices[conns[6]]].tau == conns[6].synapse.tau
    assert not filters[filter_indices[conns[6]]].is_accumulatory


def test_get_keyspace_to_filter_map(sample_network):
    """Test that a map from keyspace to filter index can be made.
    """
    (_, conns) = sample_network

    # Convert each connection to an intermediate connection with a unique
    # "keyspace".
    for c in conns:
        c.keyspace = mock.Mock()

    # Construct the minimum set of filters and provide indices from connections
    # into the set of filters.
    filters, filter_indices = filter_utils.get_combined_filters(conns)

    # Now get the map of keyspace to filter index
    filter_indices_ks = filter_utils.get_keyspace_to_filter_map(filter_indices)

    # Assert this is correct by comparing across dictionaries
    for c in conns:
        assert filter_indices[c] == filter_indices_ks[c.keyspace]


def test_get_filter_regions(sample_network):
    """Test that an appropriate filter and keys region are produced.
    """
    (_, conns) = sample_network

    # Convert each connection to an intermediate connection with a unique
    # "keyspace".
    for c in conns:
        c.keyspace = mock.Mock()

    # Assign a key and mask to each keyspace
    for i, c in enumerate(conns):
        c.keyspace.key.return_value = i
        c.keyspace.filter_mask.return_value = 0xffffffff

    # (For testing purposes)
    filters, filter_ids = filter_utils.get_combined_filters(conns)
    filter_utils.get_keyspace_to_filter_map(filter_ids)

    # Get the filter regions
    dt = 0.001
    filter_region, filter_routing_region = \
        filter_utils.get_filter_regions(conns, dt)

    # Assert the size of these regions is appropriate: Filter region should be
    # 4 * n_filters (4 words per filter) + 1 (n filters).
    assert filter_region.sizeof(slice(0, 100)) == len(filters) * 4 + 1

    # Keys region should be 4 * n_connections (4 words per connection) + 1
    assert filter_routing_region.sizeof(slice(0, 100)) == len(conns) * 4 + 1


def test_filter_region():
    """Test that the region writes out filter parameters correctly.
    """
    # Create 2 filter entries
    filters = [
        LowpassFilterParameter(1, 0.05, True, False),
        LowpassFilterParameter(2, 0.01, False, False),
    ]

    # Create the filter region
    dt = 0.01
    filter_region = filter_utils.make_filter_region(filters, dt)

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
        assert data[4 + 4*i] == f.width


def test_filter_routing_region():
    """Test that the region writes out filter routing values correctly.
    """
    # Create a mock keyspace to check that values are collected correctly.
    keyspace = mock.Mock(spec_set=['key', 'filter_mask', 'mask_d'])
    keyspace.key.return_value = 0xCAFECAFE
    keyspace.filter_mask = 0xBEEFBEEF
    keyspace.mask_d = 0xDEADABCD

    connection = mock.Mock(spec_set=['keyspace'])
    connection.keyspace = keyspace

    # Create a mapping from keyspace to filter ID
    filter_ids = {connection: 0xFEFEFFFF}

    # Create the region
    routing_region = filter_utils.make_routing_region(filter_ids)
    assert routing_region.sizeof(slice(0, 100)) == 4+1

    # Create a subregion version and ensure that the data stored within is
    # correct.
    sr = routing_region.create_subregion(slice(0, 100), 0)
    data = np.frombuffer(sr.data, dtype=np.uint32)
    assert data[0] == 1
    assert data[1:].tolist() == [keyspace.key(), keyspace.filter_mask,
                                 filter_ids[connection], keyspace.mask_d]
