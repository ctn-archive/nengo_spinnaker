import mock
import nengo
import pytest

from ..builder import Builder
from .. import builder, connection


class TestBuilderTransforms(object):
    """Check that the builder can apply transform functions.
    """
    @pytest.fixture(scope='function')
    def reset_builder(self):
        """Reset the Builder to its empty state.
        """
        Builder.network_transforms = list()
        Builder.object_transforms = dict()

    def test_network_transform(self, reset_builder):
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
        with nengo.Network() as model:
            a = nengo.Ensemble(100, 3)
            b = nengo.Ensemble(100, 3)

            with nengo.Network() as inner_model:
                c = nengo.Ensemble(100, 3)

            c1 = nengo.Connection(a, c)
            c2 = nengo.Connection(c, b)

            p = nengo.Probe(a)

        Builder.build(model, 12345)

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
