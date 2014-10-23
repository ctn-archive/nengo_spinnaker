import mock
import nengo
import numpy as np
import random

from .. import connections as connections_utils


def test_replace_object_in_connections():
    """Test that connections are replaced with new connections when they
    originate or terminate with objects which are to be replaced.
    """
    class FalseConnection(object):
        def __init__(self, pre_obj, post_obj):
            self.pre_obj = pre_obj
            self.post_obj = post_obj

        def __repr__(self):
            return "FalseConnection({}, {})".format(self.pre_obj,
                                                    self.post_obj)

    class FalseEnsemble(object):
        def __repr__(self):
            return '<FalseEnsemble at {:#x}>'.format(id(self))

    class FalseIntermediateEnsemble(object):
        def __repr__(self):
            return '<FalseIntermediateEnsemble at {:#x}>'.format(id(self))

    # Create some non-replaced objects and some replaced objects
    non_replaced_obj = [FalseEnsemble() for n in range(2)]
    replaced_objs = {FalseEnsemble(): FalseIntermediateEnsemble() for n in
                     range(3)}

    # Create a complete set of connections between these sets of objects,
    # randomly remove some elements.
    connections = list()
    connections.extend(FalseConnection(nro, ro) for nro in non_replaced_obj for
                       ro in replaced_objs.keys())
    connections.extend(FalseConnection(ro, nro) for nro in non_replaced_obj for
                       ro in replaced_objs.keys())
    random.shuffle(connections)
    connections = connections[:-3]

    # Add one connection which will remain the same
    connections.append(
        FalseConnection(non_replaced_obj[0], non_replaced_obj[1]))

    # Replace connections where objects have also been replaced
    new_connections = connections_utils.\
        replace_objects_in_connections(connections, replaced_objs)

    # Check for success
    assert len(connections) == len(new_connections)
    assert connections[-1] in new_connections

    for (oc, nc) in zip(connections[:-1], new_connections):
        assert nc is not oc

        if oc.pre_obj in replaced_objs:
            assert nc.pre_obj is replaced_objs[oc.pre_obj]
        if oc.post_obj in replaced_objs:
            assert nc.post_obj is replaced_objs[oc.post_obj]


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


def test_get_combined_connections_exact_duplicates_not_shared():
    """Exact a->b duplicates should not be shared as this would serve to halve
    the connectivity between objects.

    TODO: Increase the transform to account for this?
    """
    with nengo.Network() as model:
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)

        cs = [
            nengo.Connection(a, b),
            nengo.Connection(a, b),
        ]

    # Combine the given connections
    combined_connection_residues, combined_connection_indices = \
        connections_utils.get_combined_connections(cs)

    assert len(combined_connection_residues) == 2


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
    combined_connection_residues, combined_connection_indices = \
        connections_utils.get_combined_connections(cs)

    assert len(combined_connection_residues) == 4

    # Check 1st and 2nd combined connections
    assert combined_connection_residues[0].function is squared
    assert np.all(combined_connection_residues[0].transform == cs[0].transform)
    assert combined_connection_residues[0].keyspace is None
    assert (combined_connection_indices[cs[0]] ==
            combined_connection_indices[cs[1]] == 0)

    # Check 3rd and 4th
    assert combined_connection_indices[cs[2]] == 1
    assert combined_connection_residues[1].function is None
    assert np.all(combined_connection_residues[1].transform == cs[2].transform)
    assert combined_connection_residues[1].keyspace is None

    # Check 4th and 5th uncombined
    assert combined_connection_indices[cs[4]] == 2
    assert combined_connection_indices[cs[5]] == 3
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
    combined_connection_residues, _ = \
        connections_utils.get_combined_connections(
            connections=cs, reduced_connection_type=reduced_connection_type)

    assert len(combined_connection_residues) == 4

    # Assert reduced connection type has been called with all the connections
    for c in cs:
        assert reduced_connection_type.has_calls(c)
