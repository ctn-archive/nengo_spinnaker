"""Tests for Vertex functionality.
"""

import mock
import nengo
import numpy as np
import pytest

from nengo_spinnaker import utils


class TestMatrixRegionPartitionedByColumns(object):
    @pytest.fixture
    def matrix(self, request):
        rows = np.random.random_integers(10, 500)
        cols = np.random.random_integers(10, 500)
        return rows, cols, np.random.normal(size=(rows, cols))

    def test_shape_mismatch(self):
        # Create a test matrix
        m = np.random.normal(size=(100, 25))

        # Create the MatrixRegionPartitionedByColumns
        with pytest.raises(ValueError):
            # Test that if we specify the wrong size an error is raised
            utils.vertices.MatrixRegionPartitionedByColumns(m, shape=(101, 3))

    def test_sizeof(self):
        m = np.random.normal(size=(50, 1000))
        r = utils.vertices.MatrixRegionPartitionedByColumns(m)
        assert(r.sizeof(0, 10) == 11*50)

    def test_get_columns(self):
        m = np.random.normal(size=(50, 1000))
        r = utils.vertices.MatrixRegionPartitionedByColumns(m)
        assert(np.all(r[0:10] == m.T[0:10].T))

    def test_get_columns_random(self, matrix):
        # Create the MatrixRegionPartitionedByColumns
        rows, cols, m = matrix
        r = utils.vertices.MatrixRegionPartitionedByColumns(m)

        # Now assert that we report the size correctly
        for i in range(100):
            lo_atom = np.random.random_integers(0, cols - 5)
            hi_atom = np.random.random_integers(lo_atom + 1, cols)
            assert(np.all(r[lo_atom:hi_atom] == m.T[lo_atom:hi_atom].T))

    def test_write_to_spec(self):
        """Test that the array is written to spec file as expected.
        """
        m = np.array([[1, 2, 3], [4, 5, 6]])
        r = utils.vertices.MatrixRegionPartitionedByColumns(m)

        spec = mock.Mock()
        r.write_out(0, 1, spec)

        # Assert that the matrix is written out as an array in the correct
        # order
        assert(np.all(spec.write_array.call_args[0] ==
                      np.array([1, 2, 4, 5], dtype=np.uint32)))
        assert(spec.write_array.call_args[0][0].dtype == np.uint32)

    def test_write_to_spec_with_length(self):
        """Test that the array is written to spec file as expected.
        """
        m = np.array([[1, 2, 3], [4, 5, 6]])
        r = utils.vertices.MatrixRegionPartitionedByColumns(
            m, prepend_length=True)

        assert(r.sizeof(0, 1) == 5)

        spec = mock.Mock()
        r.write_out(0, 1, spec)

        # Assert that the matrix is written out as an array in the correct
        # order
        assert(np.all(spec.write_array.call_args[0] ==
               np.array([4, 1, 2, 4, 5], dtype=np.uint32)))
        assert(spec.write_array.call_args[0][0].dtype == np.uint32)

    def test_write_to_spec_with_formatter(self):
        """Test that the array is written to spec file as expected.
        """
        m = np.array([[1, 2, 3], [4, 5, 6]])
        r = utils.vertices.MatrixRegionPartitionedByColumns(
            m, prepend_length=True, formatter=lambda x: x**2)

        spec = mock.Mock()
        r.write_out(0, 1, spec)

        # Assert that the matrix is written out as an array in the correct
        # order and with the formatter applied to each value
        assert(np.all(spec.write_array.call_args[0] ==
                      np.array([4, 1, 4, 16, 25], dtype=np.uint32)))
        assert(spec.write_array.call_args[0][0].dtype == np.uint32)


class TestMatrixRegionPartitionedByRows(object):
    @pytest.fixture
    def matrix(self, request):
        rows = np.random.random_integers(10, 500)
        cols = np.random.random_integers(10, 500)
        return rows, cols, np.random.normal(size=(rows, cols))

    def test_shape_mismatch(self):
        # Create a test matrix
        m = np.random.normal(size=(100, 25))

        # Create the MatrixRegionPartitionedByRows
        with pytest.raises(ValueError):
            # Test that if we specify the wrong size an error is raised
            utils.vertices.MatrixRegionPartitionedByRows(m, shape=(101, 3))

    def test_sizeof(self):
        m = np.random.normal(size=(50, 1000))
        r = utils.vertices.MatrixRegionPartitionedByRows(m)
        assert(r.sizeof(0, 10) == 11*1000)

    def test_sizeof_with_length(self):
        m = np.random.normal(size=(50, 1000))
        r = utils.vertices.MatrixRegionPartitionedByRows(m, prepend_length=True)
        assert(r.sizeof(0, 10) == 11*1000 + 1)

    def test_get_rows(self):
        m = np.random.normal(size=(50, 1000))
        r = utils.vertices.MatrixRegionPartitionedByRows(m)
        assert(np.all(r[0:10] == m[0:10]))

    def test_get_rows_random(self, matrix):
        # Create the MatrixRegionPartitionedByRows
        rows, cols, m = matrix
        r = utils.vertices.MatrixRegionPartitionedByRows(m)

        # Now assert that we report the size correctly
        for i in range(100):
            lo_atom = np.random.random_integers(0, rows - 5)
            hi_atom = np.random.random_integers(lo_atom + 1, rows)
            assert(np.all(r[lo_atom:hi_atom] == m[lo_atom:hi_atom]))

    def test_write_to_spec(self):
        """Test that the array is written to spec file as expected.
        """
        m = np.array([[1, 2], [3, 4], [5, 6]])
        r = utils.vertices.MatrixRegionPartitionedByRows(m)

        spec = mock.Mock()
        r.write_out(0, 1, spec)

        # Assert that the matrix is written out as an array in the correct
        # order
        assert(np.all(spec.write_array.call_args[0] ==
                      np.array([1, 2, 3, 4], dtype=np.uint32)))
        assert(spec.write_array.call_args[0][0].dtype == np.uint32)

    def test_write_to_spec_with_length(self):
        """Test that the array is written to spec file as expected.
        """
        m = np.array([[1, 2], [3, 4], [5, 6]])
        r = utils.vertices.MatrixRegionPartitionedByRows(m, prepend_length=True)

        spec = mock.Mock()
        r.write_out(0, 1, spec)

        # Assert that the matrix is written out as an array in the correct
        # order
        assert(np.all(spec.write_array.call_args[0] ==
                      np.array([4, 1, 2, 3, 4], dtype=np.uint32)))
        assert(spec.write_array.call_args[0][0].dtype == np.uint32)

    def test_write_to_spec_with_formatter_and_length(self):
        """Test that the array is written to spec file as expected.
        """
        m = np.array([[1, 2], [3, 4], [5, 6]])
        r = utils.vertices.MatrixRegionPartitionedByRows(
            m, prepend_length=True, formatter=lambda x: x+3)

        spec = mock.Mock()
        r.write_out(0, 1, spec)

        # Assert that the matrix is written out as an array in the correct
        # order
        assert(np.all(spec.write_array.call_args[0] ==
                      np.array([4, 4, 5, 6, 7], dtype=np.uint32)))
        assert(spec.write_array.call_args[0][0].dtype == np.uint32)


class TestMakeFilterRegions(object):
    def test_basic(self):
        # Generate a simple network
        model = nengo.Network()
        with model:
            a = nengo.Ensemble(1, 1)
            b = nengo.Ensemble(1, 1)

            c1 = nengo.Connection(a, b, synapse=0.01)
            c2 = nengo.Connection(a, b, synapse=0.05)

        # Generate some keys for the connections
        ks = utils.keyspaces.create_keyspace('ks',
                                             [('o', 16), ('i', 8), ('d', 8)],
                                             'oi')
        c1k = ks(o=0, i=0)
        c2k = ks(o=0, i=1)

        # Try to build the filter regions
        (filters, routings) = utils.vertices.make_filter_regions(
            [(c1, c1k), (c2, c2k)], 0.001)

        # The filter region should be a unpartitioned list region which
        # contains exp(-dt/t), 1-exp(...) and the accumulator mask
        c1v = utils.fp.bitsk(np.exp(-0.001/0.01))
        c1v_ = utils.fp.bitsk(1. - np.exp(-0.001/0.01))
        c2v = utils.fp.bitsk(np.exp(-0.001/0.05))
        c2v_ = utils.fp.bitsk(1. - np.exp(-0.001/0.05))
        expected_filters = [c1v, c1v_, 0xffffffff,
                            c2v, c2v_, 0xffffffff]
        expected_routings = [c1k.routing_key, c1k.routing_mask, 0, 0xff,
                             c2k.routing_key, c2k.routing_mask, 1, 0xff]

        assert(filters.data == expected_filters)
        assert(routings.data == expected_routings)
