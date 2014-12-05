import mock
import numpy as np
import random

from .. import connections as connections_utils
from ..connections import (get_pre_padded_transform,
                           get_keyspaces_with_dimensions)
from ...connections.reduced import OutgoingReducedConnection


def test_replace_object_in_connections():
    """Test that connections are replaced with new connections when they
    originate or terminate with objects which are to be replaced.
    """
    class FalseConnection(object):
        def __init__(self, pre_obj, post_obj):
            self.pre_obj = pre_obj
            self.post_obj = post_obj

        def __repr__(self):
            return "FalseConnection({}, {})".format(self.pre_obj,
                                                    self.post_obj)

    class FalseEnsemble(object):
        def __repr__(self):
            return '<FalseEnsemble at {:#x}>'.format(id(self))

    class FalseIntermediateEnsemble(object):
        def __repr__(self):
            return '<FalseIntermediateEnsemble at {:#x}>'.format(id(self))

    # Create some non-replaced objects and some replaced objects
    non_replaced_obj = [FalseEnsemble() for n in range(2)]
    replaced_objs = {FalseEnsemble(): FalseIntermediateEnsemble() for n in
                     range(3)}

    # Create a complete set of connections between these sets of objects,
    # randomly remove some elements.
    connections = list()
    connections.extend(FalseConnection(nro, ro) for nro in non_replaced_obj for
                       ro in replaced_objs.keys())
    connections.extend(FalseConnection(ro, nro) for nro in non_replaced_obj for
                       ro in replaced_objs.keys())
    random.shuffle(connections)
    connections = connections[:-3]

    # Add one connection which will remain the same
    connections.append(
        FalseConnection(non_replaced_obj[0], non_replaced_obj[1]))

    # Replace connections where objects have also been replaced
    new_connections = connections_utils.\
        replace_objects_in_connections(connections, replaced_objs)

    # Check for success
    assert len(connections) == len(new_connections)
    assert connections[-1] in new_connections

    for (oc, nc) in zip(connections[:-1], new_connections):
        assert nc is not oc

        if oc.pre_obj in replaced_objs:
            assert nc.pre_obj is replaced_objs[oc.pre_obj]
        if oc.post_obj in replaced_objs:
            assert nc.post_obj is replaced_objs[oc.post_obj]


class TestPadTransform(object):
    """Test conversions from pre-slice and transform into full transform
    matrices.
    """
    def test_non_sliced(self):
        """Test getting padded matrices when there is no slicing.
        """
        pre_slice = slice(None)
        transform = np.array([[1, 2, 3], [4, 5, 6]])
        size_in = 3

        # Assert that we just get the transform back
        new_transform = get_pre_padded_transform(pre_slice, size_in, transform)
        assert new_transform is not transform
        assert np.all(new_transform == transform)

    def test_sliced(self):
        """Test getting padded matrices when there is slicing but a full matrix
        is provided.
        """
        pre_slice = slice(0, 2)
        transform = np.array([[1, 2], [2, 3], [4, 5]])
        size_in = 5

        # Assert that the new transform is correct
        new_transform = get_pre_padded_transform(pre_slice, size_in, transform)
        assert np.all(
            new_transform == np.array([[1, 2, 0, 0, 0],
                                       [2, 3, 0, 0, 0],
                                       [4, 5, 0, 0, 0]])
        )

    def test_sliced_discontiguous(self):
        """Test getting padded matrices when there is slicing but a full matrix
        is provided.
        """
        pre_slice = slice(0, 3, 2)
        transform = np.array([[1, 2], [2, 3], [4, 5]])
        size_in = 5

        # Assert that the new transform is correct
        new_transform = get_pre_padded_transform(pre_slice, size_in, transform)
        assert np.all(
            new_transform == np.array([[1, 0, 2, 0, 0],
                                       [2, 0, 3, 0, 0],
                                       [4, 0, 5, 0, 0]])
        )

    def test_scalar(self):
        """Test getting a matrix when there is no slicing but only a scalar is
        provided.
        """
        pre_slice = slice(None)
        transform = 3.
        size_in = 5

        # Assert that the new transform is correct
        assert np.all(
            get_pre_padded_transform(pre_slice, size_in, transform) ==
            np.eye(5) * 3.
        )

    def test_sliced_scalar(self):
        """Test getting padded matrices when there is slicing and only a scalar
        is provided.
        """
        pre_slice = slice(2, 4)
        transform = 3.
        size_in = 5

        # Assert that the new transform is correct
        new_transform = get_pre_padded_transform(pre_slice, size_in, transform)
        assert np.all(
            new_transform == np.array([[0, 0, 3., 0, 0],
                                       [0, 0, 0, 3., 0]])
        )

    def test_sliced_scalar_discontiguous(self):
        """Test getting padded matrices when there is slicing and only a scalar
        is provided.
        """
        pre_slice = slice(0, 5, 2)
        transform = 3.
        size_in = 5

        # Assert that the new transform is correct
        new_transform = get_pre_padded_transform(pre_slice, size_in, transform)
        assert np.all(
            new_transform == np.array([[3., 0., 0., 0., 0.],
                                       [0., 0., 3., 0., 0.],
                                       [0., 0., 0., 0., 3.]])
        )


class TestGetKeyspacesWithDimensions(object):
    """Test that outgoing keyspaces can be generated for connections
    with/without slicing.
    """
    def test_get_keyspaces_with_dimensions_no_slice(self):
        """Get that the keyspace is called appropriately for the unsliced
        connection here.
        """
        # Create a mock keyspace
        ks = mock.Mock()
        dim_ks = {d: mock.Mock(name='Keyspace1 d={}'.format(d)) for d in
                  range(5)}
        ks.side_effect = lambda n_dimension: dim_ks[n_dimension]

        ks2 = mock.Mock()
        dim_ks2 = {d: mock.Mock(name='Keyspace2 d={}'.format(d)) for d in
                   range(3)}
        ks2.side_effect = lambda n_dimension: dim_ks2[n_dimension]

        # Create a reduced outgoing connection
        orcs = [
            OutgoingReducedConnection(
                width=5, transform=1., function=None, pre_slice=slice(None),
                post_slice=slice(None), keyspace=ks),
            OutgoingReducedConnection(
                width=3, transform=1., function=None, pre_slice=slice(None),
                post_slice=slice(None), keyspace=ks2)
        ]

        # Get the keyspaces with dimensions included
        keyspaces = get_keyspaces_with_dimensions(orcs)

        # Assert that the keyspace was called correctly
        ks.assert_has_calls([mock.call(n_dimension=d) for d in range(5)])
        ks2.assert_has_calls([mock.call(n_dimension=d) for d in range(3)])

        # Assert the returned keys are in the correct order
        assert keyspaces == ([dim_ks[d] for d in range(5)] +
                             [dim_ks2[d] for d in range(3)])
