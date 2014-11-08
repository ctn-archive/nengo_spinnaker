import collections
import mock
import nengo
from six import itervalues

from ..connection_tree import ConnectionTree
from ..intermediate import IntermediateConnection
from ..reduced import (
    OutgoingReducedConnection, StandardInputPort,
)
from ...spinnaker.edges import Edge


def test_from_intermediate_connections():
    """Test that a new ConnectionTree can be built from intermediate connection
    objects.

    - Tests correct return of originating/terminating/all objects
    - Tests correct return of outgoing/incoming connections for objects
    """
    # Create some mock objects
    obj_a = mock.Mock(name='a')
    obj_a.size_in = 3
    obj_b = mock.Mock(name='b')
    obj_b.size_in = 3
    obj_c = mock.Mock(name='c')
    obj_c.size_in = 3

    default_keyspace = mock.Mock()

    # Create some connections between these objects
    # obj_a should have 1 outgoing connection and 1 incoming
    # obj_b should have 0 outgoing connections and 2 incoming
    # obj_c should have 1 outgoing connection and 1 incoming
    conns = [
        IntermediateConnection(obj_a, obj_b, slice(None), slice(None),
                               synapse=nengo.Lowpass(0.2),
                               keyspace=default_keyspace),
        IntermediateConnection(obj_a, obj_c, slice(None), slice(None),
                               synapse=nengo.Lowpass(0.1),
                               keyspace=default_keyspace),
        IntermediateConnection(obj_a, obj_a, slice(None), slice(None),
                               synapse=nengo.Lowpass(0.1),
                               keyspace=default_keyspace),
        IntermediateConnection(obj_c, obj_b, slice(None), slice(None),
                               synapse=nengo.Lowpass(0.1),
                               function=lambda x: 2*x,
                               keyspace=default_keyspace),
    ]

    # Create the connection tree
    tree = ConnectionTree.from_intermediate_connections(conns)

    # Assert that the tree is correct:
    # Originating objects
    assert len(tree.get_originating_objects()) == 2
    assert set(tree.get_originating_objects()) == set([obj_a, obj_c])

    # Terminating objects
    assert len(tree.get_terminating_objects()) == 3
    assert set(tree.get_terminating_objects()) == set([obj_a, obj_b, obj_c])

    # All objects
    assert len(tree.get_objects()) == 3
    assert set(tree.get_objects()) == set([obj_a, obj_b, obj_c])

    # Outgoing connections for obj_a
    assert len(tree.get_outgoing_connections(obj_a)) == 1
    assert (list(tree.get_outgoing_connections(obj_a)) ==
            [OutgoingReducedConnection(3, 1.0, None,
                                       slice(None), slice(None),
                                       keyspace=default_keyspace)])

    # Outgoing connections obj_b
    assert len(tree.get_outgoing_connections(obj_b)) == 0
    assert list(tree.get_outgoing_connections(obj_b)) == list()

    # Outgoing connections obj_c
    assert len(tree.get_outgoing_connections(obj_c)) == 1
    assert (list(tree.get_outgoing_connections(obj_c))[0] ==
            OutgoingReducedConnection(3, 1.0, conns[-1].function,
                                      slice(None), slice(None),
                                      keyspace=default_keyspace))

    # Incoming connections, arranged by port, then by filter
    for obj, l in [(obj_a, 1), (obj_b, 2), (obj_c, 1)]:
        incs = tree.get_incoming_connections(obj)
        assert list(incs.keys()) == [StandardInputPort]
        assert len(incs[StandardInputPort]) == l

        for kss in itervalues(incs[StandardInputPort]):
            for ks in kss:
                assert ks is default_keyspace


def test_replace_objects():
    """Tests that a new ConnectionTree can be built from an old one by
    replacing objects.  """
    # Create some objects
    obj_a = mock.Mock(name='a')
    obj_a.size_in = 3
    obj_b = mock.Mock(name='b')
    obj_b.size_in = 4

    # Create some connections between these objects
    conns = [
        IntermediateConnection(obj_a, obj_b, slice(None), slice(None),
                               nengo.Lowpass(0.3)),
        IntermediateConnection(obj_a, obj_a, slice(None), slice(None),
                               nengo.Lowpass(0.3)),
    ]

    # Create a new connection tree
    tree = ConnectionTree.from_intermediate_connections(conns)

    # Replace obj_a with obj_c
    obj_c = mock.Mock(name='c')
    new_tree = tree.get_new_tree_with_replaced_objects({obj_a: obj_c})

    # Assert that the new tree is NOT the old tree
    assert new_tree is not tree

    # Check that all the connections have been copied rather than referenced by
    # the new tree, perform a parallel walk of the tree.
    for (obj, new_obj) in [(obj_a, obj_c), (obj_b, obj_b)]:
        # Get the output connections for the objects, assert that connection
        # objects are equivalent but not the same object.
        old_conns = tree.get_outgoing_connections(obj)
        new_conns = new_tree.get_outgoing_connections(new_obj)

        for (c, new_c) in zip(old_conns, new_conns):
            assert c == new_c and c is not new_c

        # Get the incoming connections for the objects, assert that the
        # dictionaries are equivalent but that the contained objects are
        # different.
        old_conns = tree.get_incoming_connections(obj)
        new_conns = new_tree.get_incoming_connections(new_obj)

        assert list(old_conns.keys()) == list(new_conns.keys())
        for (ok, nk) in zip(list(old_conns.keys()), list(new_conns.keys())):
            old_iconns = old_conns[ok]
            new_iconns = new_conns[nk]

            for (a, b) in zip(old_iconns.keys(), new_iconns.keys()):
                assert a == b and a is not b
                assert old_iconns[a] == new_iconns[b]


def test_apply_keyspace():
    """Tests that a new ConnectionTree can be built from an old one by applying
    a keyspace and filling in the values of existing keyspaces.
    """
    # Create a mock keyspace that will be applied to the output connections of
    # a connection tree.
    keyspace = mock.Mock(name="DefaultKeyspace", spec_set=['__call__'])
    obj_keyspaces = {
        0: mock.Mock(name="Object0Keyspace", spec_set=['__call__']),
        1: mock.Mock(name="Object1Keyspace", spec_set=['__call__']),
    }
    keyspace.side_effect = lambda o: obj_keyspaces[o]

    obj0_conn_keyspaces = {
        0: mock.Mock(name="Object0Connection0", spec_set=['__call__']),
        1: mock.Mock(name="Object0Connection1", spec_set=['__call__']),
    }
    obj_keyspaces[0].side_effect = lambda i: obj0_conn_keyspaces[i]

    obj1_conn_keyspaces = {
        0: mock.Mock(name="Object1Connection0", spec_set=['__call__']),
        1: mock.Mock(name="Object1Connection1", spec_set=['__call__']),
    }
    obj_keyspaces[1].side_effect = lambda i: obj1_conn_keyspaces[i]

    # Create objects to be the referents in connections
    a = mock.Mock(name='A', spec_set=['size_in'])
    a.size_in = 3
    b = mock.Mock(name='B', spec_set=['size_in'])
    b.size_in = 3
    c = mock.Mock(name='C', spec_set=['size_in'])
    c.size_in = 3

    # Create some connections
    conns = [
        IntermediateConnection(a, c, function=lambda x: x),
        IntermediateConnection(a, c, function=lambda x: 2*x),
        IntermediateConnection(b, c),
        IntermediateConnection(b, c, function=lambda x: x),
    ]
    for c in conns:
        c.synapse = nengo.Lowpass(0.02)

    # Build the first connection tree
    tree = ConnectionTree.from_intermediate_connections(conns)

    # Now build the new connection tree, check that the correct arguments were
    # passed to the keyspaces.
    new_tree = tree.get_new_tree_with_applied_keyspace(keyspace)

    # Check the top-level keyspace
    keyspace.assert_has_calls([mock.call(o=0), mock.call(o=1)])
    assert keyspace.call_count == 2

    # Check the keyspaces for objects
    for i in range(2):
        obj_keyspaces[i].assert_has_calls([mock.call(i=0), mock.call(i=1)])
        assert obj_keyspaces[i].call_count == 2

    # Build a mapping of keyspace to connection and check that it as we expect,
    # simultaneously check that the connection sets are disjunct.
    keyspace_to_conn = collections.defaultdict(list)

    assert set(new_tree.get_objects()) == set(tree.get_objects())

    for obj in new_tree.get_objects():
        for conn in new_tree.get_outgoing_connections(obj):
            keyspace_to_conn[conn.keyspace].append(conn)

            # Check this actual connection is not present in the old tree
            for old_conn in tree.get_outgoing_connections(obj):
                assert old_conn is not conn

    assert set(keyspace_to_conn.keys()) == (set(obj0_conn_keyspaces.values()) |
                                            set(obj1_conn_keyspaces.values()))
    for conns in list(keyspace_to_conn.values()):
        assert len(conns) == 1


def test_fold_connections():
    """Test that a connectivity tree can be folded to form just a list of
    connections between objects.
    """
    # Create some objects
    obj_a = mock.Mock(name='a')
    obj_a.size_in = 3
    obj_b = mock.Mock(name='b')
    obj_b.size_in = 4

    keyspace0 = mock.Mock(name='Keyspace0')
    keyspace1 = mock.Mock(name='Keyspace1')

    # Create some connections between these objects
    conns = [
        IntermediateConnection(obj_a, obj_b, slice(None), slice(None),
                               nengo.Lowpass(0.3), keyspace=keyspace0),
        IntermediateConnection(obj_a, obj_a, slice(None), slice(None),
                               nengo.Lowpass(0.3), keyspace=keyspace1),
    ]

    # Create a new connection tree
    tree = ConnectionTree.from_intermediate_connections(conns)

    # Fold the tree
    conns = tree.get_folded_edges()
    for c in conns:
        if c.post_vertex == obj_b:
            assert c == Edge(obj_a, obj_b, keyspace0)
        else:
            assert c == Edge(obj_a, obj_a, keyspace1)
