import collections
import mock
import nengo
import pytest

from ..builder import Builder
from .. import builder, connection


@pytest.fixture(scope='function')
def sample_model():
    with nengo.Network() as model:
        a = nengo.Ensemble(100, 3, label='a')
        b = nengo.Ensemble(100, 3, label='b')

        with nengo.Network() as inner_model:
            c = nengo.Ensemble(100, 3, label='c')

        c1 = nengo.Connection(a, c)
        c2 = nengo.Connection(c, b)

        p = nengo.Probe(a)

    return (model, (a, b, c), (c1, c2), (p,))


class TestBuilderTransforms(object):
    """Check that the builder can apply transform functions.
    """
    @pytest.fixture(scope='function')
    def reset_builder(self):
        """Reset the Builder to its empty state.
        """
        Builder.network_transforms = list()
        Builder.object_transforms = dict()

    def test_network_transform(self, reset_builder, sample_model):
        """Test that a network transform can be registered and will be applied
        to the network when building.  Also tests that networks are properly
        reduced.
        """
        # Create a fake network transform
        network_transform = mock.Mock()
        network_transform.side_effect = lambda os, cs, ps: (os, cs)

        # Register the transform and ensure that the builder calls it
        Builder.add_network_transform(network_transform)

        # Create a fake network to ensure that objects are passed along
        model, (a, b, c), (c1, c2), (p,) = sample_model
        Builder.build(model)

        # Assert that the network transform was called with appropriate
        # arguments.
        assert network_transform.call_count == 1
        called_objs = network_transform.call_args[0][0]
        called_conns = network_transform.call_args[0][1]
        called_probes = network_transform.call_args[0][2]

        # Check that all the objects were present
        assert sorted(called_objs) == sorted([a, b, c])

        # Check that all the connections were present
        assert sorted(called_conns) == sorted([c1, c2])

        # Check that the probe was present
        assert called_probes == [p]

    def test_object_build(self, reset_builder):
        """Test that objects are built correctly."""
        # Create types which we can try to build
        class A(object):
            pass

        class B(A):
            pass

        class C(object):
            pass

        class D(A, C):  # Bigger MRO
            pass

        a_builder = mock.Mock()

        # Register the object builder
        Builder.add_object_builder(A, a_builder)

        # Now try to build an instance of each object
        a = A()
        Builder.build_obj(a)
        a_builder.assert_called_with(a)

        b = B()
        Builder.build_obj(b)
        a_builder.assert_called_with(b)

        c = C()
        d = Builder.build_obj(c)  # No builder, pass through
        assert d is c

        d = D()
        Builder.build_obj(d)
        a_builder.assert_called_with(d)


def test_convert_remaining_connections():
    """Test that only Nengo Connection objects are replaced.
    """
    with nengo.Network():
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)
        c1 = nengo.Connection(a, b)
    c2 = mock.Mock()

    new_conns = builder._convert_remaining_connections([c1, c2])
    assert new_conns[1] is c2
    assert new_conns[0].pre_obj is a
    assert new_conns[0].post_obj is b


def test_build_keyspace(sample_model):
    """Assert that an appropriate keyspace is built.
    """
    # Get the model and build the connection tree
    model, (a, b, c), (c1, c2), (p,) = sample_model
    new_conns = builder._convert_remaining_connections([c1, c2])
    tree = connection.build_connection_trees(new_conns)

    # Build the keyspace
    ks = builder._build_keyspace(tree, subobject_bits=7)
    print ks.__field_lengths__
    assert ks.__field_lengths__['x'] == 1  # eXternal bound packet
    assert ks.__field_lengths__['o'] == 1  # Object index
    assert ks.__field_lengths__['s'] == 7  # Sub-object index
    assert ks.__field_lengths__['i'] == 1  # Connection Index
    assert ks.__field_lengths__['d'] == 2  # Dimension index
    assert ks.__routing_fields__ == 'xosi'
    assert ks.__filter_fields__ == 'xoi'


def test_apply_default_keyspace(sample_model):
    # Get the model and build the connection tree
    model, (a, b, c), (c1, c2), (p,) = sample_model
    new_conns = builder._convert_remaining_connections([c1, c2])

    # Create the connection tree
    tree = connection.build_connection_trees(new_conns)

    # Create mock keyspaces
    ks = mock.Mock()
    core_ks = {
        0: mock.Mock(),
        1: mock.Mock(),
    }
    c0c0 = mock.Mock()
    c1c0 = mock.Mock()
    ks.side_effect = lambda o: core_ks[o]
    core_ks[0].return_value = c0c0
    core_ks[1].return_value = c1c0

    # Apply default keyspaces, ensuring the calls are appropriate
    new_tree = builder._apply_default_keyspace(ks, tree)

    assert ks.call_count == 2
    assert core_ks[0].call_count == 1
    assert core_ks[1].call_count == 1

    # Check the new tree to ensure that keyspaces are assigned only once
    assigned_ks = collections.defaultdict(list)
    for conns in new_tree.values():
        for c in conns:
            assigned_ks[c.keyspace].append(c)

    assert len(assigned_ks) == 2
    assert len(assigned_ks[c0c0]) == 1
    assert len(assigned_ks[c1c0]) == 1

    # Assert that a new tree has been returned...
    assert tree is not new_tree


def test_apply_default_keyspace_ignore_given_ks(sample_model):
    # Get the model and build the connection tree
    model, (a, b, c), (c1, c2), (p,) = sample_model
    new_conns = builder._convert_remaining_connections([c1, c2])

    # Add a new connection with a given keyspace to ensure that it is not
    # overwritten.
    preks_with_o = mock.Mock(name='ExistingKeyspaceWithO')
    preks = mock.Mock(name='ExistingKeyspace')
    preks.return_value = preks_with_o
    preks.is_set_o = False
    new_conns.append(
        connection.IntermediateConnection(b, c, synapse=nengo.Lowpass(0.05),
                                          keyspace=preks)
    )

    # Create the connection tree
    tree = connection.build_connection_trees(new_conns)

    # Create mock keyspace
    ks = mock.Mock()

    # Apply default keyspaces, ensuring the calls are appropriate
    new_tree = builder._apply_default_keyspace(ks, tree)

    # Check that the pre-provided keyspace was called correctly.
    assert any([mock.call(o=i) in preks.mock_calls for i in range(3)])

    # Check the new tree to ensure that keyspaces are assigned only once
    assigned_ks = collections.defaultdict(list)
    for conns in new_tree.values():
        for c in conns:
            assigned_ks[c.keyspace].append(c)

    assert len(assigned_ks[preks_with_o]) == 1

    # Assert that a new tree has been returned...
    assert tree is not new_tree


def test_apply_default_keyspace_ignore_given_ks_no_fill(sample_model):
    # Get the model and build the connection tree
    model, (a, b, c), (c1, c2), (p,) = sample_model
    new_conns = builder._convert_remaining_connections([c1, c2])

    # Add a new connection with a given keyspace to ensure that it is not
    # overwritten.
    preks = mock.Mock(name='ExistingKeyspace')
    preks.is_set_o = True
    new_conns.append(
        connection.IntermediateConnection(b, c, synapse=nengo.Lowpass(0.05),
                                          keyspace=preks)
    )

    # Create the connection tree
    tree = connection.build_connection_trees(new_conns)

    # Create mock keyspace
    ks = mock.Mock()

    # Apply default keyspaces, ensuring the calls are appropriate
    new_tree = builder._apply_default_keyspace(ks, tree)

    # Check that the pre-provided keyspace was called correctly.
    assert not preks.called

    # Check the new tree to ensure that keyspaces are assigned only once
    assigned_ks = collections.defaultdict(list)
    for conns in new_tree.values():
        for c in conns:
            assigned_ks[c.keyspace].append(c)

    assert len(assigned_ks[preks]) == 1

    # Assert that a new tree has been returned...
    assert tree is not new_tree


def test_replace_objects_in_connection_trees():
    """Create a mock connection tree and assert that objects are replaced
    correctly.
    """
    # Create a mock tree, add some objects and assert that they are replaced
    # correctly.
    tree = collections.defaultdict(lambda: collections.defaultdict(list))

    a = mock.Mock(name='A')
    b = mock.Mock(name='B')

    oc = connection.OutgoingReducedConnection(1.0, None)
    ic = connection.IncomingReducedConnection(connection.Target(b), None)

    tree[a][oc].append(ic)

    # Replacement objects
    new_a = mock.Mock(name='New A')
    new_b = mock.Mock(name='New B')
    repls = {a: new_a, b: new_b}

    # Replace the objects
    new_tree = builder._replace_objects_in_connection_trees(repls, tree)

    # Assert new tree is correct
    assert new_tree is not tree
    assert len(new_tree) == 1
    assert len(new_tree[new_a]) == 1
    new_oc = new_tree[new_a].keys()[0]
    assert len(new_tree[new_a][new_oc]) == 1
    new_ic = new_tree[new_a][new_oc][0]
    assert new_ic.target.target_object is new_b

    assert tree[a][oc][0].target.target_object is b  # Check the original
