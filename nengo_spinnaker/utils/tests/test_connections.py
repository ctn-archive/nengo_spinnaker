import mock
import nengo
import numpy as np

from .. import connections as connections_utils


def test_TransformFunctionKeyspaceConnection():
    """Test that connections are correctly reduced.
    """
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)

        c = nengo.Connection(a, b)

    c1 = connections_utils.TransformFunctionKeyspaceConnection(c)

    assert np.all(c1.transform == c.transform)
    assert c1.function == c.function
    assert c1.keyspace is None


def test_get_combined_connections():
    """Test that connections can be correctly combined.
    """
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)
        c = nengo.Ensemble(100, 1)

        squared = lambda x: x**2
        cs = [
            nengo.Connection(a, b, function=squared),
            nengo.Connection(a, c, function=squared),  # Shared
            nengo.Connection(a, b, transform=1.5),
            nengo.Connection(a, c, transform=1.5),  # Shared
            nengo.Connection(a, b, transform=0.9, function=lambda x: 2*x),
            nengo.Connection(a, c, transform=0.8, function=lambda x: 2*x),
        ]

    # Combine the given connections
    combined_connection_residues = \
        connections_utils.get_combined_connections(cs)

    assert len(combined_connection_residues) == 4

    # Check 1st and 2nd combined connections
    assert combined_connection_residues[0].function is squared
    assert np.all(combined_connection_residues[0].transform == cs[0].transform)
    assert combined_connection_residues[0].keyspace is None

    # Check 3rd and 4th
    assert combined_connection_residues[1].function is None
    assert np.all(combined_connection_residues[1].transform == cs[2].transform)
    assert combined_connection_residues[1].keyspace is None

    # Check 4th and 5th uncombined
    assert np.all(combined_connection_residues[2].transform == cs[4].transform)
    assert np.all(combined_connection_residues[3].transform == cs[5].transform)


def test_get_combined_connections_custom_func():
    """Test that connections can be correctly combined.
    """
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)
        c = nengo.Ensemble(100, 1)

        squared = lambda x: x**2
        cs = [
            nengo.Connection(a, b, function=squared),
            nengo.Connection(a, c, function=squared),  # Shared
            nengo.Connection(a, b, transform=1.5),
            nengo.Connection(a, c, transform=1.5),  # Shared
            nengo.Connection(a, b, transform=0.9, function=lambda x: 2*x),
            nengo.Connection(a, c, transform=0.8, function=lambda x: 2*x),
        ]

    # Custom residue type
    reduced_connection_type = mock.Mock()
    reduced_connection_type.side_effect = \
        connections_utils.TransformFunctionKeyspaceConnection

    # Combine the given connections
    combined_connection_residues = \
        connections_utils.get_combined_connections(
            connections=cs, reduced_connection_type=reduced_connection_type)

    assert len(combined_connection_residues) == 4

    # Assert reduced connection type has been called with all the connections
    for c in cs:
        assert reduced_connection_type.has_calls(c)
