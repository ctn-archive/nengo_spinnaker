import mock
import pytest

from .. import assembler
from ..spinnaker import vertices


class TestAssembler(object):
    @pytest.fixture(scope='function')
    def reset_assembler_objects(self):
        assembler.Assembler.object_builders = dict()
        assembler.Assembler.connection_builders = dict()

    def test_register_object_builder(self, reset_assembler_objects):
        """Check that a new object builder can be registered and used.
        """
        # Add a new object builder
        build_fn = mock.Mock()
        build_fn.return_value = None

        assembler.Assembler.register_object_builder(build_fn, mock.Mock)

        # Check that it can be called
        obj = mock.Mock()
        asmblr = assembler.Assembler()
        asmblr.build_object(obj)

        build_fn.assert_called_once_with(obj, asmblr)

    def test_register_object_builder_vertex(self, reset_assembler_objects):
        """Check that a new object builder can be registered and used.
        """
        # Add a new object builder
        build_fn = mock.Mock()
        build_fn.return_value = vertices.NengoVertex(None, None)

        assembler.Assembler.register_object_builder(build_fn, mock.Mock)

        # Check that it can be called
        obj = mock.Mock()
        asmblr = assembler.Assembler()
        asmblr.time_in_seconds = 5.
        vertex = asmblr.build_object(obj)

        # Assert that the time in seconds is passed on
        assert vertex.runtime == asmblr.time_in_seconds

    def test_object_build_fail(self, reset_assembler_objects):
        """Assert a TypeError is raised if we try to build an object that
        isn't recognised.
        """
        asmblr = assembler.Assembler()
        with pytest.raises(TypeError):
            asmblr.build_object(mock.Mock())

    def test_connection_builder_fail(self, reset_assembler_objects):
        """Check that a TypeError is raised if a connection cannot be built.
        """
        # Create a dummy connection to try building
        connection = mock.Mock()
        connection.pre_obj = mock.Mock()
        connection.post_obj = mock.Mock()

        # Ensure that building this fails
        asmblr = assembler.Assembler()

        with pytest.raises(TypeError):
            asmblr.build_connection(connection)

    def test_connection_builder_fn(self, reset_assembler_objects):
        """Check that a TypeError is raised if a connection cannot be built.
        """
        # Register a new connection builder to build connections from Mock to
        # Mock
        connection_fn = mock.Mock()
        assembler.Assembler.register_connection_builder(
            connection_fn, mock.Mock, mock.Mock)

        # Create a dummy connection to try building
        connection = mock.Mock()
        connection.pre_obj = mock.Mock()
        connection.post_obj = mock.Mock()

        # Ensure that building this calls the function with the correct
        # arguments
        asmblr = assembler.Assembler()
        asmblr.build_connection(connection)
        connection_fn.assert_called_once_with(connection, asmblr)

    def test_assemble_vertex(self):
        """Test that the vertex assemble function just returns the vertex it
        was given.
        """
        vertex = vertices.NengoVertex(None, None)
        rval = assembler.vertex_builder(vertex, mock.Mock())

        assert rval is vertex

    def test_assemble_node(self):
        """Test that the node assemble function doesn't do anything.
        """
        rval = assembler.assemble_node(mock.Mock(), mock.Mock())
        assert rval is None

    def test_call(self, reset_assembler_objects):
        class TestObject(object):
            pass

        a = TestObject()
        b = TestObject()
        c = TestObject()

        objs = [a, b, c]
        obj_verts = {k: vertices.NengoVertex(None, None) for k in objs[:-1]}
        obj_verts.update({c: None})

        def build_test_object(test_object, asmblr):
            return obj_verts[test_object]

        class TestConnection(object):
            def __init__(self, a, b):
                self.pre_obj = a
                self.post_obj = b

        conns = [
            TestConnection(a, b),
            TestConnection(b, c),
        ]

        def build_test_connection(test_connection, assembler):
            return mock.Mock()

        assembler.Assembler.register_object_builder(
            build_test_object, TestObject)
        assembler.Assembler.register_connection_builder(
            build_test_connection)

        # Construct a sample network and call an assembler
        asmblr = assembler.Assembler()
        vs, es = asmblr([a, b, c], conns, 5.0, 0.001)

        # Assert that utility functions work
        for obj in obj_verts:
            assert asmblr.get_object_vertex(obj) is obj_verts[obj]

        assert asmblr.get_incoming_connections(b) == [conns[0]]
        assert asmblr.get_outgoing_connections(b) == [conns[1]]
