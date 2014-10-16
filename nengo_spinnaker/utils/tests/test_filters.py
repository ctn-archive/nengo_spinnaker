import nengo

from .. import filters as filter_utils


def test_get_combined_filters():
    """Test that filters may be combined correctly.
    """
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
        conns = [
            nengo.Connection(a, b, synapse=0.01),
            nengo.Connection(a, b, transform=0.1, synapse=0.01),
            nengo.Connection(a, b, synapse=0.05),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.05)),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.01)),
            nengo.Connection(a, b, synapse=nengo.synapses.Alpha(0.01)),
            nengo.Connection(a, b, synapse=0.01),
        ]

        setattr(conns[-1], 'is_accumulatory', True)

    # Construct the minimum set of filters and provide indices from connections
    # into the set of filters.
    filters, filter_indices = filter_utils.get_combined_filters(conns)

    # Assert that there are the correct number of filters and that the
    # connection indices are appropriate.
    assert len(filters) == 5
    assert filter_indices[conns[0]] == filter_indices[conns[1]]
    assert filter_indices[conns[2]] not in [
        i for (c, i) in filter_indices.items() if c is not conns[2]]
    assert filter_indices[conns[3]] not in [
        i for (c, i) in filter_indices.items() if c is not conns[3]]
    assert filter_indices[conns[4]] == filter_indices[conns[5]]
    assert filter_indices[conns[6]] not in [
        i for (c, i) in filter_indices.items() if c is not conns[6]]

    # Assert that parameters have been grabbed correctly
    assert filters[filter_indices[conns[0]]].tau == conns[0].synapse.tau
    assert not filters[filter_indices[conns[0]]].is_accumulatory
    assert filters[filter_indices[conns[2]]].tau == conns[2].synapse.tau
    assert not filters[filter_indices[conns[2]]].is_accumulatory
    assert filters[filter_indices[conns[3]]].tau == conns[3].synapse.tau
    assert not filters[filter_indices[conns[3]]].is_accumulatory
    assert filters[filter_indices[conns[4]]].tau == conns[4].synapse.tau
    assert not filters[filter_indices[conns[4]]].is_accumulatory
    assert filters[filter_indices[conns[6]]].tau == conns[6].synapse.tau
    assert filters[filter_indices[conns[6]]].is_accumulatory
