import copy
import mock
import nengo
import numpy as np
import pytest

from ..builder import Builder
from .. import builder
from ..connections.connection_tree import ConnectionTree


@pytest.fixture(scope='function')
def reset_builder(request):
    """Reset the Builder to its empty state.
    """
    old_net_transforms = Builder.network_transforms[:]
    old_obj_transforms = copy.copy(Builder.object_transforms)

    Builder.network_transforms = list()
    Builder.object_transforms = dict()

    def restore():
        Builder.network_transforms = old_net_transforms
        Builder.object_transforms = old_obj_transforms
    request.addfinalizer(restore)


@pytest.fixture(scope='function')
def sample_model():
    with nengo.Network() as model:
        a = nengo.Ensemble(100, 3, label='a')
        b = nengo.Ensemble(100, 3, label='b')

        a.eval_points = np.random.uniform(size=(100, 3))
        b.eval_points = np.random.uniform(size=(100, 3))

        with nengo.Network():
            c = nengo.Ensemble(100, 3, label='c')
            c.eval_points = np.random.uniform(size=(100, 3))

        c1 = nengo.Connection(a, c)
        c2 = nengo.Connection(c, b)

        p = nengo.Probe(a)

    return (model, (a, b, c), (c1, c2), (p,))


class TestBuilderTransforms(object):
    """Check that the builder can apply transform functions.
    """
    def test_network_transform(self, reset_builder, sample_model):
        """Test that a network transform can be registered and will be applied
        to the network when building.  Also tests that networks are properly
        reduced.
        """
        # Create a fake network transform
        network_transform = mock.Mock()
        network_transform.side_effect = lambda os, cs, ps, ss: (os, cs)

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
        called_seeds = network_transform.call_args[0][3]

        # Check that all the objects were present
        assert sorted(called_objs) == sorted([a, b, c])

        # Check that all the connections were present
        assert sorted(called_conns) == sorted([c1, c2])

        # Check that the probe was present
        assert called_probes == [p]

        # Check seeds were present for all objects
        for x in (a, b, c):
            assert x in called_seeds

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
        c_trees = mock.Mock(name='Connection tree')
        config = mock.Mock(name='Config')
        seed = mock.Mock(name='Seed')

        a = A()
        Builder.build_obj(a, c_trees, config, seed)
        a_builder.assert_called_with(a, c_trees, config, seed)

        b = B()
        Builder.build_obj(b, c_trees, config, seed)
        a_builder.assert_called_with(b, c_trees, config, seed)

        # No builder, pass through
        c = C()
        d = Builder.build_obj(c, c_trees, config, seed)
        assert d is c

        d = D()
        Builder.build_obj(d, c_trees, config, seed)
        a_builder.assert_called_with(d, c_trees, config, seed)

    def test_object_build_gets_seeds(self, reset_builder):
        with nengo.Network() as model:
            a = nengo.Ensemble(100, 1)
            b = nengo.Ensemble(100, 1)
            a.eval_points = b.eval_points = np.random.uniform(size=(100, 1))
            nengo.Connection(a, b)

        # Create and add a mock ensemble builder
        ens_builder = mock.Mock()
        ens_builder.side_effect = lambda obj, cs, conf, seeds: obj
        Builder.add_object_builder(nengo.Ensemble, ens_builder)

        # Build the network, and then ensure that the same seeds are passed
        Builder.build(model)
        assert ens_builder.call_count == 2
        assert (ens_builder.call_args_list[0][0][3] ==
                ens_builder.call_args_list[1][0][3])
        assert set(ens_builder.call_args_list[0][0][3].keys()) == set([a, b])

    def test_decorators(self, reset_builder):
        """Check that creating functions with the Builder decorators adds them
        to the list of methods to apply.
        """
        @Builder.network_transform
        def test_network_transform(objs, conns, probes):
            raise NotImplementedError

        assert [test_network_transform] == Builder.network_transforms

        @Builder.object_builder(mock.Mock)
        def test_object_builder(obj, *args):
            raise NotImplementedError

        assert {mock.Mock: test_object_builder} == Builder.object_transforms


def test_get_seed():
    """Test utility for getting seeds.
    """
    rng = mock.Mock(spec_set=['randint'])
    obj1 = mock.Mock(spec_set=['seed'])
    obj2 = mock.Mock(spec_set=[])

    assert builder._get_seed(obj1, rng) is obj1.seed
    rng.randint.assert_called_with(np.iinfo(np.int32).max)

    builder._get_seed(obj2, rng)
    assert rng.randint.call_count == 2


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
    tree = ConnectionTree.from_intermediate_connections(new_conns)

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


def test_runthrough():
    # As a final test check that we can actually run through the builder in its
    # unmodified state...
    with nengo.Network(label='test') as model:
        a = nengo.Ensemble(100, 2, label='a')
        b = nengo.Ensemble(100, 1, label='b')

        nengo.Connection(a[0], b, function=lambda x: x**2)

    conn_tree, rngs = Builder.build(model)  # noqa
