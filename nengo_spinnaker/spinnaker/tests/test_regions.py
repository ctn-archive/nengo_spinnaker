import mock
import numpy as np
import pytest

from .. import regions


# Matrix region tests
class TestMatrixRegion(object):
    """Tests for the MatrixRegion and derived types."""

    def test_init_no_shape(self):
        """Test that a MatrixRegion can not be created if not shape or matrix
        is specified.
        """
        with pytest.raises(Exception):
            regions.MatrixRegion()

    def test_init_mismatched_shape(self):
        """Test that a MatrixRegion can not be created if the specified shape
        and matrix do not match.
        """
        with pytest.raises(AssertionError):
            regions.MatrixRegion(np.array([1]), shape=(1, 2))

    def test_matrix_sizeof(self):
        """Test that the size of a matrix is correctly returned, in this case
        the region is unpartitioned."""
        mr = regions.MatrixRegion(np.zeros(100))
        assert mr.sizeof(slice(0, 50)) == 100

    def test_matrix_sizeof_with_prepends(self):
        mr = regions.MatrixRegion(np.zeros(100), prepends=[
                regions.MatrixRegionPrepends.N_ATOMS,
                regions.MatrixRegionPrepends.N_ROWS,
                regions.MatrixRegionPrepends.N_COLUMNS,
                regions.MatrixRegionPrepends.SIZE])
        assert mr.sizeof(slice(0, 50)) == 100 + 4

    def test_matrix_sizeof_oversized(self):
        mr = regions.MatrixRegion(np.zeros(100))
        assert mr.sizeof(slice(0, 101)) == 100

    def test_matrix_sizeof_undersized(self):
        mr = regions.MatrixRegion(np.zeros(100))
        assert mr.sizeof(slice(0, 10)) == 100

    def test_matrix_rows_sizeof(self):
        mr = regions.MatrixRegionPartitionedByRows(np.zeros((100, 5)))
        assert mr.sizeof(slice(0, 10)) == 5*10

    def test_matrix_columns_sizeof(self):
        mr = regions.MatrixRegionPartitionedByColumns(np.zeros((15, 100)))
        assert mr.sizeof(slice(0, 10)) == 10*15

    def test_create_subregion(self):
        data = np.array([[n]*5 for n in range(10)], dtype=np.uint32)
        mr = regions.MatrixRegionPartitionedByRows(data)
        sr = mr.create_subregion(slice(0, 10), 0)

        assert np.all(
            np.frombuffer(sr.data, dtype=np.uint32).reshape(data.shape) ==
            data[0:10])
        assert sr.size_words == 10*5
        assert not sr.unfilled

    def test_create_subregion_with_formatter(self):
        data = np.array([[n]*5 for n in range(10)], dtype=np.uint32)
        mr = regions.MatrixRegionPartitionedByRows(data,
                                                   formatter=lambda x: x**2)
        sr = mr.create_subregion(slice(0, 10), 0)

        assert np.all(
            np.frombuffer(sr.data, dtype=np.uint32).reshape(data.shape) ==
            data[0:10] ** 2)
        assert sr.size_words == 10*5
        assert not sr.unfilled

    def test_create_subregion_with_prepend_n_atoms(self):
        data = np.array([[n]*5 for n in range(100)], dtype=np.uint32)
        mr = regions.MatrixRegionPartitionedByRows(data, prepends=[
            regions.MatrixRegionPrepends.N_ATOMS])
        sr = mr.create_subregion(slice(0, 10), 0)

        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 10
        assert np.all(sr_data[1:].reshape(data[0:10].shape) == data[0:10])
        assert sr.size_words == 10*5 + 1
        assert not sr.unfilled

    def test_create_subregion_with_prepend_full_length(self):
        data = np.array([[n]*5 for n in range(100)], dtype=np.uint32)
        mr = regions.MatrixRegionPartitionedByRows(data, prepends=[
            regions.MatrixRegionPrepends.SIZE])
        sr = mr.create_subregion(slice(0, 10), 0)

        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 50
        assert np.all(sr_data[1:].reshape(data[0:10].shape) == data[0:10])
        assert sr.size_words == 10*5 + 1
        assert not sr.unfilled

    def test_create_subregion_with_prepend_full_length_n_atoms(self):
        data = np.array([[n]*5 for n in range(100)], dtype=np.uint32)
        mr = regions.MatrixRegionPartitionedByRows(data, prepends=[
            regions.MatrixRegionPrepends.N_ATOMS,
            regions.MatrixRegionPrepends.SIZE])
        sr = mr.create_subregion(slice(0, 10), 0)

        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 10
        assert sr_data[1] == 50
        assert np.all(sr_data[2:].reshape(data[0:10].shape) == data[0:10])
        assert sr.size_words == 10*5 + 2
        assert not sr.unfilled

    def test_size_no_matrix(self):
        mr = regions.MatrixRegion(shape=(100, 5, 2), prepends=[
            regions.MatrixRegionPrepends.N_ATOMS,
            regions.MatrixRegionPrepends.SIZE])
        assert mr.sizeof(slice(0, 10)) == 100*5*2 + 2

    def test_size_no_matrix_rows(self):
        mr = regions.MatrixRegionPartitionedByRows(shape=(100, 5), prepends=[
            regions.MatrixRegionPrepends.N_ATOMS,
            regions.MatrixRegionPrepends.SIZE])
        assert mr.sizeof(slice(0, 10)) == 10*5 + 2

    def test_size_no_matrix_columns(self):
        mr = regions.MatrixRegionPartitionedByColumns(
            shape=(100, 5), prepends=[
                regions.MatrixRegionPrepends.N_ATOMS,
                regions.MatrixRegionPrepends.SIZE
            ])
        assert mr.sizeof(slice(0, 2)) == 100*2 + 2


# Keys Region Tests
class TestKeysRegion(object):
    @pytest.fixture(scope='function')
    def keys(self):
        # Create a set of keys
        keys = [mock.Mock(spec_set=['key']) for i in range(12)]
        for i, k in enumerate(keys):
            k.key.return_value = i
        return keys

    @pytest.fixture(scope='function')
    def keys_with_routing(self):
        # Create a set of keys
        keys = [mock.Mock(spec_set=['key', 'routing_mask']) for i in range(12)]
        for i, k in enumerate(keys):
            k.key.return_value = i
            k.routing_mask = 0xFFEEFFEE
        return keys

    def test_fill_in_field(self, keys):
        # Create a new key region
        r = regions.KeysRegion(keys, fill_in_field='i')

        # Get the size of the region
        assert r.sizeof(slice(0, 10)) == len(keys)

        # Create a subregion and ensure that the keys are used correctly
        sr = r.create_subregion(slice(0, 10), 1)

        for k in keys:
            k.key.assert_called_once_with(**{'i': 1})

        # Assert that a Subregion with the correct data is returned
        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        for i, k in enumerate(keys):
            assert sr_data[i] == k.key.return_value

    def test_subregion_n_entries(self, keys_with_routing):
        keys = keys_with_routing

        # Create a new key region
        r = regions.KeysRegion(keys, fill_in_field='i',
                               extra_fields=[lambda k, i: k.routing_mask],
                               prepend_n_keys=True)

        # Get the size of the region
        assert r.sizeof(slice(0, 10)) == 2*len(keys) + 1

        # Create a subregion and ensure that the keys are used correctly
        sr = r.create_subregion(slice(0, 10), 1)

        # Assert that a Subregion with the correct data is returned
        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 12

    def test_extra_fields(self, keys_with_routing):
        keys = keys_with_routing

        # Create a new key region
        r = regions.KeysRegion(keys, fill_in_field='i', extra_fields=[
                               lambda k, i: k.routing_mask])

        # Get the size of the region
        assert r.sizeof(slice(0, 10)) == len(keys)*2

        # Create a subregion and ensure that the keys are used correctly
        sr = r.create_subregion(slice(0, 10), 1)

        # Assert that a Subregion with the correct data is returned
        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        for i, d in enumerate(sr_data[0::2]):
            assert d == i
        for d in sr_data[1::2]:
            assert d == 0xFFEEFFEE

    def test_with_partitioning(self, keys):
        # Create a new keys region that is partitioned
        r = regions.KeysRegion(keys, partitioned=True, prepend_n_keys=True,
                               fill_in_field='i')

        # Assert sizing is correct
        assert r.sizeof(slice(0, 10)) == 10 + 1
        assert r.sizeof(slice(3, 7)) == 4 + 1

        # Create a subregion and ensure that the keys are used correctly
        vslice = slice(3,7)
        sr = r.create_subregion(vslice, 1)

        for k in keys[vslice]:
            k.key.assert_called_once_with(**{'i': 1})

        # Assert that a Subregion with the correct data is returned
        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data.size == (vslice.stop - vslice.start) + 1
        assert sr_data[0] == vslice.stop - vslice.start
        for i, k in enumerate(keys[vslice]):
            assert sr_data[i+1] == k.key.return_value

    def test_with_partitioning_and_extra_fields(self, keys):
        # Create a new keys region that is partitioned
        r = regions.KeysRegion(keys, partitioned=True, prepend_n_keys=True,
                               fill_in_field='i', extra_fields=[
                               lambda k, i: i])

        # Assert sizing is correct
        assert r.sizeof(slice(0, 10)) == 2*10 + 1
        assert r.sizeof(slice(3, 7)) == 2*4 + 1

        # Create a subregion and ensure that the keys are used correctly
        vslice = slice(3,7)
        sr = r.create_subregion(vslice, 0xFEED)

        for k in keys[vslice]:
            k.key.assert_called_once_with(**{'i': 0xFEED})

        # Assert that a Subregion with the correct data is returned
        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data.size == 2*(vslice.stop - vslice.start) + 1
        assert sr_data[0] == vslice.stop - vslice.start
        for i, k in enumerate(keys[vslice]):
            assert sr_data[1+2*i] == k.key.return_value
            assert sr_data[1+2*i+1] == 0xFEED
