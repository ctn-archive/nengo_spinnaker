import copy
import mock
import nengo
import pytest

from ..assembler import Assembler
from ..connections.connection_tree import ConnectionTree
from ..connections.intermediate import IntermediateConnection
from ..spinnaker.edges import Edge


class FakeObject(object):
    def __init__(self, size_in=3):
        self.size_in = size_in


class DerivedFromFakeObject(FakeObject):
    pass


@pytest.fixture(scope="function")
def reset_assembler(request):
    """Fixture to return the Assembler to a blank state."""
    prior_assemblers = copy.copy(Assembler.object_assemblers)
    Assembler.object_assemblers = dict()

    def restore():
        Assembler.object_assemblers = prior_assemblers
    request.addfinalizer(restore)


@pytest.fixture(scope="function")
def sample_ctree(request):
    # Now create a connection tree that contains a simple connection between
    # two objects of this type.
    obj_a = DerivedFromFakeObject()
    obj_b = DerivedFromFakeObject()
    conns = [
        IntermediateConnection(obj_a, obj_b, synapse=nengo.Lowpass(0.05)),
    ]
    return ConnectionTree.from_intermediate_connections(conns), (obj_a, obj_b)


class TestAssembler(object):
    def test_add_object_assembler(self, reset_assembler, sample_ctree):
        """Test that a new assembler can be registered and is called."""
        ctree, (obj_a, obj_b) = sample_ctree

        # Create a new object assembler
        def assemble_fn(obj, connection_trees, config, rngs, runtime, dt,
                        machine_timestep):
            pass

        # Wrap it so that we can check the calls
        assemble = mock.Mock(wraps=assemble_fn)

        # Register the assembler
        Assembler.add_object_assembler(FakeObject, assemble)

        # Call the assemble object function to check parameters are passed
        # through correctly, and that the MRO is used to find the correct
        # assembly function.
        config = mock.Mock()
        rngs = dict()
        runtime = 10.0
        dt = 0.001
        machine_timestep = 1000

        Assembler.assemble_obj(obj_a, ctree, config, rngs, runtime, dt,
                               machine_timestep)

        # Assert the function was called with the given parameters
        assemble.assert_called_once_with(obj_a, ctree, config, rngs, runtime,
                                         dt, machine_timestep)

    def test_object_assembler_decorator(self, reset_assembler):
        """Test that a decorator can be used to add new assembler functions."""
        # Get the assemblers prior to adding a new function
        prior_assemblers = copy.copy(Assembler.object_assemblers)

        # Use the decorator
        @Assembler.object_assembler(DerivedFromFakeObject)
        def assembler(*args):
            pass

        # Check that the class has been registered with the correct build
        # function.
        assert DerivedFromFakeObject not in prior_assemblers
        assert DerivedFromFakeObject in Assembler.object_assemblers
        assert Assembler.object_assemblers[DerivedFromFakeObject] is assembler

    def test_assemble_obj_unknown(self, reset_assembler):
        """Check that the assembler returns items it has no assembler for."""
        obj = mock.Mock()
        obj_after_assembly = Assembler.assemble_obj(obj, None, None, None, 0.,
                                                    0.001, 1000)
        assert obj is obj_after_assembly

    def test_assemble(self, reset_assembler, sample_ctree):
        """Test that the assembler can assemble from a connection tree."""
        ctree, (obj_a, obj_b) = sample_ctree

        # Create a new assembler
        new_objs = {
            obj_a: FakeObject(),
            obj_b: FakeObject(),
        }
        assemble = mock.Mock()
        assemble.side_effect = lambda obj, *args: new_objs[obj]

        # Register the assembler
        Assembler.add_object_assembler(DerivedFromFakeObject, assemble)

        # Call the assemble object function to check parameters are passed
        # through correctly, and that the MRO is used to find the correct
        # assembly function.
        config = mock.Mock()
        rngs = dict()
        runtime = 10.0
        dt = 0.001
        machine_timestep = 1000

        objs, edges = Assembler.assemble(ctree, config, rngs, runtime, dt,
                                         machine_timestep)

        # Assert the function was called with the given parameters
        assemble.assert_has_calls([
            mock.call(obj_a, ctree, config, rngs, runtime, dt,
                      machine_timestep),
            mock.call(obj_b, ctree, config, rngs, runtime, dt,
                      machine_timestep),
        ], any_order=True)

        # Assert the new connection tree includes the new objects, and that the
        # connections are still as they were.
        assert set(objs) == set(new_objs.values())
        assert edges == [Edge(new_objs[obj_a], new_objs[obj_b], None)]
