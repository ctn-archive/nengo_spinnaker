import mock
import nengo
import pytest

from .. import connection


# Tests for the connection builder
class TestGenericConnectionBuilder(object):
    @pytest.fixture
    def network(self):
        model = nengo.Network()
        with model:
            a = nengo.Ensemble(100, 1)
            b = nengo.Ensemble(100, 1)

            c = nengo.Connection(a, b)

        c1 = connection.IntermediateConnection.from_connection(c)
        return c1, c, a, b

    def test_build(self, network):
        # Create a Nengo connection
        c1, c, a, b = network
        c1.keyspace = mock.Mock()

        # Create a mock assembler
        # This assembler will return different objects for pre and post
        assembler = mock.Mock()
        prev = mock.Mock()
        postv = mock.Mock()
        assembler.get_object_vertex.side_effect = \
            lambda obj: prev if obj is a else postv

        # Call the builder function
        ne = connection.generic_connection_builder(c1, assembler)

        # Assert that the correct methods were called on the assembler
        assembler.get_object_vertex.assert_any_call(a)
        assembler.get_object_vertex.assert_any_call(b)

        # Assert that the correct pre-vertex and post-vertex objects are
        # present
        assert ne.pre_vertex is prev
        assert ne.post_vertex is postv
        assert ne.keyspace is c1.keyspace

    def test_no_keyspace(self, network):
        # Create a Nengo connection
        c1, c, a, b = network

        # Create a mock assembler
        # This assembler will return different objects for pre and post
        assembler = mock.Mock()
        prev = mock.Mock()
        postv = mock.Mock()
        assembler.get_object_vertex.side_effect = \
            lambda obj: prev if obj is a else postv

        # Call the builder function
        with pytest.raises(AssertionError):
            connection.generic_connection_builder(c1, assembler)

    def test_no_prevertex(self, network):
        # Create a Nengo connection
        c1, c, a, b = network

        # Create a mock assembler
        # This assembler will return different objects for pre and post
        assembler = mock.Mock()
        prev = None
        postv = mock.Mock()
        assembler.get_object_vertex.side_effect = \
            lambda obj: prev if obj is a else postv

        # Call the builder function
        ne = connection.generic_connection_builder(c1, assembler)
        assert ne is None

    def test_no_postvertex(self, network):
        # Create a Nengo connection
        c1, c, a, b = network

        # Create a mock assembler
        # This assembler will return different objects for pre and post
        assembler = mock.Mock()
        prev = mock.Mock()
        postv = None
        assembler.get_object_vertex.side_effect = \
            lambda obj: prev if obj is a else postv

        # Call the builder function
        ne = connection.generic_connection_builder(c1, assembler)
        assert ne is None
