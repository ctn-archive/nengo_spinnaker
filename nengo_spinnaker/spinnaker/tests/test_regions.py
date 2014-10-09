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

    def test_matrix_sizeof_with_full_length(self):
        mr = regions.MatrixRegion(np.zeros(100), prepend_full_length=True)
        assert mr.sizeof(slice(0, 50)) == 101

    def test_matrix_sizeof_with_n_atoms(self):
        mr = regions.MatrixRegion(np.zeros(100), prepend_n_atoms=True)
        assert mr.sizeof(slice(0, 50)) == 101

    def test_matrix_sizeof_with_n_atoms_and_full_length(self):
        mr = regions.MatrixRegion(np.zeros(100), prepend_n_atoms=True,
                                  prepend_full_length=True)
        assert mr.sizeof(slice(0, 50)) == 102

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
        mr = regions.MatrixRegionPartitionedByRows(data, prepend_n_atoms=True)
        sr = mr.create_subregion(slice(0, 10), 0)

        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 10
        assert np.all(sr_data[1:].reshape(data[0:10].shape) == data[0:10])
        assert sr.size_words == 10*5 + 1
        assert not sr.unfilled

    def test_create_subregion_with_prepend_full_length(self):
        data = np.array([[n]*5 for n in range(100)], dtype=np.uint32)
        mr = regions.MatrixRegionPartitionedByRows(data,
                                                   prepend_full_length=True)
        sr = mr.create_subregion(slice(0, 10), 0)

        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 50
        assert np.all(sr_data[1:].reshape(data[0:10].shape) == data[0:10])
        assert sr.size_words == 10*5 + 1
        assert not sr.unfilled

    def test_create_subregion_with_prepend_full_length_n_atoms(self):
        data = np.array([[n]*5 for n in range(100)], dtype=np.uint32)
        mr = regions.MatrixRegionPartitionedByRows(data, prepend_n_atoms=True,
                                                   prepend_full_length=True)
        sr = mr.create_subregion(slice(0, 10), 0)

        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 10
        assert sr_data[1] == 50
        assert np.all(sr_data[2:].reshape(data[0:10].shape) == data[0:10])
        assert sr.size_words == 10*5 + 2
        assert not sr.unfilled

    def test_size_no_matrix(self):
        mr = regions.MatrixRegion(shape=(100, 5, 2), prepend_n_atoms=True,
                                  prepend_full_length=True)
        assert mr.sizeof(slice(0, 10)) == 100*5*2 + 2

    def test_size_no_matrix_rows(self):
        mr = regions.MatrixRegionPartitionedByRows(shape=(100, 5),
                                                   prepend_n_atoms=True,
                                                   prepend_full_length=True)
        assert mr.sizeof(slice(0, 10)) == 10*5 + 2

    def test_size_no_matrix_columns(self):
        mr = regions.MatrixRegionPartitionedByColumns(shape=(100, 5),
                                                      prepend_n_atoms=True,
                                                      prepend_full_length=True)
        assert mr.sizeof(slice(0, 2)) == 100*2 + 2


# Keys Region Tests
class TestKeysRegion(object):
    def test_fill_in_field(self):
        # Create a set of keys
        keys = [mock.Mock() for i in range(12)]
        for i, k in enumerate(keys):
            k.key.return_value = i

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

    def test_subregion_n_atoms(self):
        # Create a set of keys
        keys = [mock.Mock() for i in range(12)]
        for i, k in enumerate(keys):
            k.key.return_value = i

        # Create a new key region
        r = regions.KeysRegion(keys, fill_in_field='i', prepend_n_atoms=True)

        # Get the size of the region
        assert r.sizeof(slice(0, 10)) == len(keys) + 1

        # Create a subregion and ensure that the keys are used correctly
        sr = r.create_subregion(slice(0, 10), 1)

        # Assert that a Subregion with the correct data is returned
        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 10

    def test_subregion_full_length(self):
        # Create a set of keys
        keys = [mock.Mock() for i in range(12)]
        for i, k in enumerate(keys):
            k.key.return_value = i

        # Create a new key region
        r = regions.KeysRegion(keys, fill_in_field='i',
                               prepend_full_length=True)

        # Get the size of the region
        assert r.sizeof(slice(0, 10)) == len(keys) + 1

        # Create a subregion and ensure that the keys are used correctly
        sr = r.create_subregion(slice(0, 10), 1)

        # Assert that a Subregion with the correct data is returned
        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 12

    def test_subregion_full_length_n_atoms(self):
        # Create a set of keys
        keys = [mock.Mock() for i in range(12)]
        for i, k in enumerate(keys):
            k.key.return_value = i

        # Create a new key region
        r = regions.KeysRegion(keys, fill_in_field='i', prepend_n_atoms=True,
                               prepend_full_length=True)

        # Get the size of the region
        assert r.sizeof(slice(0, 10)) == len(keys) + 2

        # Create a subregion and ensure that the keys are used correctly
        sr = r.create_subregion(slice(0, 10), 1)

        # Assert that a Subregion with the correct data is returned
        sr_data = np.frombuffer(sr.data, dtype=np.uint32)
        assert sr_data[0] == 10
        assert sr_data[1] == 12

    def test_extra_fields(self):
        # Create a set of keys
        keys = [mock.Mock() for i in range(12)]
        for i, k in enumerate(keys):
            k.key.return_value = i
            k.routing_mask = 0xFFFFEEEE

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
            assert d == 0xFFFFEEEE
