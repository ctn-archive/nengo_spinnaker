import mock
import nengo
import nengo.utils.builder
import numpy as np
import pytest

from .. import builder, connection


class TestBuilder(object):
    @pytest.fixture(scope='function')
    def reset_builder(self):
        builder.Builder.pre_rpn_transforms = list()
        builder.Builder.post_rpn_transforms = list()

    @pytest.fixture(scope='function')
    def sample_network(self):
        model = nengo.Network()
        with model:
            a = nengo.Node(lambda t: 0.5*t, size_out=1, size_in=0, label='a')
            b = nengo.Ensemble(100, 4, label='b')
            c = nengo.Ensemble(100, 4, label='c')
            d = nengo.Node(lambda t, x: None, size_in=4, size_out=0, label='d')
            e = nengo.Node(lambda t, x: None, size_in=4, size_out=0, label='e')

            nengo.Connection(a, b[0])
            nengo.Connection(a, c[0])
            nengo.Connection(b[0], b[1:], transform=[[0.33]]*3)
            nengo.Connection(b[0], b[0], transform=0.)

            nengo.Connection(b, d, solver=nengo.solvers.Lstsq())
            nengo.Connection(b, e, solver=nengo.solvers.LstsqNoise())

        objs, conns = nengo.utils.builder.objs_and_connections(model)
        return model, objs, conns

    def test_register_connectivity_transform(self, reset_builder,
                                             sample_network):
        """Assert that a new connectivity transform can be constructed and is
        called correctly.
        """
        model, objs, conns = sample_network

        # Create a sample connectivity transform
        sample_connectivity_transform = mock.Mock()
        sample_connectivity_transform.side_effect = lambda os, cs, ps: (os, cs)

        # Register it
        builder.Builder.register_connectivity_transform(
            sample_connectivity_transform)

        # Build the network and assert that the correct arguments are passed to
        # the connectivity transform.
        builder.Builder.build(model, 0.001, 1)

        assert sample_connectivity_transform.called_once_with(objs, conns, [])

    def test_register_object_transform(self, reset_builder, sample_network):
        """Assert that a new "object" transform can be registered and called.
        """
        model, objs, conns = sample_network

        # Create a sample object transform
        sample_object_transform = mock.Mock()
        sample_object_transform.side_effect = \
            lambda os, cs, ps, dt, rng: (os, cs)

        # Register it
        builder.Builder.register_object_transform(sample_object_transform)

        # Build the network and assert that the correct arguments are passed to
        # the connectivity transform.
        dt = 0.001
        builder.Builder.build(model, dt, 1)

        # All connections should have been replaced with an equivalent
        rep_conns = [connection.IntermediateConnection.from_connection(c) for
                     c in conns]
        args = sample_object_transform.call_args[0]
        assert args[0] == objs

        assert len(rep_conns) == len(args[1])

        for (a, b) in zip(args[1], rep_conns):
            assert ((a.pre_obj is b.pre_obj) and
                     (a.post_obj is b.post_obj) and
                     (a.synapse == b.synapse) and
                     (a.function == b.function) and
                     (np.all(a.transform == b.transform)) and
                     (a.solver == b.solver) and
                     (a.width == b.width) and
                     (a.is_accumulatory == b.is_accumulatory) and
                     (a.learning_rule == b.learning_rule) and
                     (a.modulatory == b.modulatory))

        assert args[2] == list()
        assert args[3] == dt

    def test_get_outgoing_ids(self, sample_network):
        """Assert that unique connection IDs are assigned to each connection
        from each object.
        """
        model, objs, conns = sample_network

        # Get connection indices
        connection_indices = builder._get_outgoing_ids(conns)

        # Assert that all keys are present
        for conn in conns:
            assert conn in connection_indices

        # Assert that shared connections are shared
        assert connection_indices[conns[-2]] != connection_indices[conns[-1]]
