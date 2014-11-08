import mock
import nengo
import numpy as np

from ...assembler import Assembler
from ...connections.connection_tree import ConnectionTree
from ...connections.intermediate import IntermediateConnection
from ...connections.reduced import OutgoingReducedConnection
from ..value_filter import (
    make_filter_system_region, make_transform_region_data,
    make_transform_region, ValueFilter
)
from ...utils.fixpoint import bitsk


def test_create_filter_vertex():
    """Create a network including a filter object and ensure that an
    appropriate vertex is built.
    """
    # Create objects to go in the connection tree
    obj_a = mock.Mock(name='Object A', spec_set=[])
    obj_b = mock.Mock(name='Object B', spec_set=['size_in'])
    obj_b.size_in = 4

    f = ValueFilter(3, 10, 1)

    # Build a connection tree with connections A -> f, f -> B
    ks = mock.Mock(name='Keyspace')
    a_to_f = IntermediateConnection(obj_a, f, synapse=nengo.Lowpass(0.05),
                                    keyspace=ks)
    f_to_b = IntermediateConnection(f, obj_b, synapse=nengo.Lowpass(0.0),
                                    keyspace=ks)
    ctree = ConnectionTree.from_intermediate_connections([a_to_f, f_to_b])

    # Build the filter
    filter_vertex = Assembler.assemble_obj(f, ctree, None, {}, 10., .001, 1000)

    # Assert that it is sensible
    assert all([filter_vertex is not x for x in [None, f]])
    assert len(filter_vertex.regions) == 5

    # Assert that usage can be reported
    filter_vertex.get_resources_used_by_atoms(slice(10), None)


def test_make_filter_system_region():
    """Check that a filter region can be made and read correctly."""
    size_in = 3
    size_out = 5
    machine_timestep = 100
    transmission_delay = 7
    interpacket_pause = 1

    # Make the filter region
    filter_region = make_filter_system_region(
        size_in=size_in, size_out=size_out, machine_timestep=machine_timestep,
        transmission_delay=transmission_delay,
        interpacket_pause=interpacket_pause
    )

    # Check sizing
    assert filter_region.sizeof(slice(0, 100)) == 5
    assert filter_region.sizeof(slice(2, 5)) == 5

    # Check partitioning and data
    sr = filter_region.create_subregion(slice(0, 1), 0)
    sr_data = np.frombuffer(sr.data, dtype=np.uint32).tolist()
    assert sr_data == [size_in, size_out, machine_timestep, transmission_delay,
                       interpacket_pause]


def test_make_transform_region_data():
    """Check that a transform region can be made and read correctly."""
    # Create a list of outgoing connections
    orcs = [
        OutgoingReducedConnection(
            width=2, transform=3., function=None, pre_slice=slice(None),
            post_slice=slice(None)),
        OutgoingReducedConnection(
            width=3, transform=np.eye(2)*2., function=None,
            pre_slice=slice(1, 3), post_slice=slice(None)),
        OutgoingReducedConnection(
            width=1, transform=np.array([[1., 3.]]), function=None,
            pre_slice=slice(0, 4, 2), post_slice=slice(None)),
        OutgoingReducedConnection(
            width=4, transform=np.array([[2., 3.]]*4), function=None,
            pre_slice=slice(0, 2), post_slice=slice(None)),
    ]

    correct_transform = np.array([[3., 0., 0.],  # orc[0]
                                  [0., 3., 0.],  # ""
                                  [0., 0., 3.],  # ""
                                  [0., 2., 0.],  # orc[1]
                                  [0., 0., 2.],  # ""
                                  [1., 0., 3.],  # orc[2]
                                  [2., 3., 0.],  # orc[3]
                                  [2., 3., 0.],  # ""
                                  [2., 3., 0.],  # ""
                                  [2., 3., 0.]])
    fp_correct_transform = np.array(bitsk(correct_transform), dtype=np.uint32)

    # Get the full transform
    transform = make_transform_region_data(orcs, size_in=3)
    assert np.all(transform == correct_transform)

    # Get the region
    transform_region = make_transform_region(orcs, size_in=3)

    # Check how the region partitions and returns its data
    sr1 = transform_region.create_subregion(slice(0, 4), 0)
    sr1_data = np.frombuffer(sr1.data, dtype=np.uint32).reshape(4, 3)
    assert np.all(sr1_data == fp_correct_transform[0:4])
