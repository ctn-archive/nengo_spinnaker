import mock
import pytest

from .. import assembler
from ..spinnaker import vertices


class TestAssembler(object):
    @pytest.fixture(scope='function')
    def reset_assembler_objects(self):
        assembler.Assembler.object_builders = dict()

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
