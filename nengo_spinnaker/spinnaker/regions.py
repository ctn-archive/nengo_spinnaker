import collections
import enum
import numpy as np


class Region(object):
    """Generic memory region object.

    Attributes
    ----------
    in_dtcm : bool
        Whether the region of memory is loaded into DTCM.
    unfilled : bool
        Region is left unfilled.
    """
    def __init__(self, in_dtcm=True, unfilled=False):
        """Create a new region.

        Parameters
        ----------
        in_dtcm : bool
            Whether the region is stored in DTCM.
        unfilled : bool
            Whether the region is to be left unfilled.
        formatter : func
            Formatting function to apply to each element in the array before
            writing out.
        """
        self.in_dtcm = in_dtcm
        self.unfilled = unfilled

    def sizeof(self, vertex_slice):
        """Get the size (in words) of the region.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms that will be represented by the region.
        """
        raise NotImplementedError

    def create_subregion(self, vertex_slice, subvertex_index):
        """Create a smaller version of the region ready to write to memory.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms that will be represented by the region.
        """
        raise NotImplementedError


class Subregion(collections.namedtuple('Subregion',
                                       'data size_words unfilled')):
    def __new__(cls, data, size_words, unfilled):
        if data is not None:
            assert data.dtype == np.uint32  # May want to revise this later
            d = np.copy(data)
            d.flags.writeable = False
            data = d.data
        return super(cls, Subregion).__new__(cls, data, size_words, unfilled)


MatrixRegionPrepends = enum.Enum('MatrixRegionPrepends',
                                 'N_ATOMS N_ROWS N_COLUMNS SIZE INDEX')


class MatrixRegion(Region):
    """An unpartitioned region of memory representing a matrix.
    """
    partition_index = None  # Not partitioned along any axis

    def __init__(self, matrix=None, shape=None, dtype=np.uint32, in_dtcm=True,
                 unfilled=False, prepends=list(), formatter=None):
        """Create a new region representing a matrix.

        Parameters
        ----------
        matrix : ndarray
            Matrix to represent in this region.
        shape : tuple
            Shape of the matrix, will be taken from the passed matrix if not
            specified.
        """
        super(MatrixRegion, self).__init__(in_dtcm=in_dtcm, unfilled=unfilled)
        self.prepends = prepends
        self.formatter = formatter

        # Assert the matrix matches the given shape
        assert shape is not None or matrix is not None

        if shape is not None and matrix is not None:
            assert shape == matrix.shape

        if shape is None and matrix is not None:
            shape = matrix.shape

        # Store the matrix and shape
        self.matrix = matrix
        self.shape = shape
        self.dtype = dtype

    def sizeof(self, vertex_slice):
        """Get the size of the region for the specified atoms in words.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms that will be represented by the region.
        """
        return (self.size_from_shape(vertex_slice) + len(self.prepends))

    def __getitem__(self, index):
        return self.matrix

    def size_from_shape(self, vertex_slice):
        """Get the size from the shape of the matrix.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms that will be represented by the region.
        """
        # If the shape is n-D then multiply the length of the axes together,
        # accounting for the clipping of the partitioned axis.
        vertex_slice = vertex_slice.as_slice
        return reduce(
            lambda x, y: x*y,
            [s if i != self.partition_index else
             (min(s, vertex_slice.stop) - max(0, vertex_slice.start)) for
             i, s in enumerate(self.shape)])

    def create_subregion(self, vertex_slice, subvertex_index):
        """Return the data to write to memory for this region.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms that will be represented by the region.
        subvertex_index : int
            The index of the subvertex containing this region.
        """
        # Get the data, flatten
        data = self[vertex_slice.as_slice]
        flat_data = data.reshape(data.size)

        # Format the data as required
        if self.formatter is None:
            formatted_data = flat_data
        else:
            formatted_data = np.array(map(self.formatter, flat_data.tolist()),
                                      dtype=np.uint32)

        # Prepend any required additional data
        prepend_data = {
            MatrixRegionPrepends.N_ATOMS: vertex_slice.n_atoms,
            MatrixRegionPrepends.N_ROWS: data.shape[0],
            MatrixRegionPrepends.N_COLUMNS: (0 if len(data.shape) == 1 else
                                             data.shape[1]),
            MatrixRegionPrepends.SIZE: data.size,
            MatrixRegionPrepends.INDEX: subvertex_index,
        }
        prepends = np.array([prepend_data[p] for p in self.prepends],
                            dtype=self.dtype)

        srd = np.hstack([prepends, formatted_data])
        return Subregion(srd, len(srd), self.unfilled)


class MatrixRegionPartitionedByColumns(MatrixRegion):
    """A region representing a matrix which is partitioned by columns.
    """
    partition_index = 1

    def __getitem__(self, index):
        return self.matrix.T[index].T


class MatrixRegionPartitionedByRows(MatrixRegion):
    """A region representing a matrix which is partitioned by rows.
    """
    partition_index = 0

    def __getitem__(self, index):
        return self.matrix[index]


class KeysRegion(Region):
    """A region which represents a series of keys.

    A region which represents keys to be used in transmitting or receiving
    multicast packets.  If desired a field may be filled in with the subvertex
    index when a subregion is created.

    This region may be partitioned by passing the `partitioned` parameter.  If
    this is done then one key is written out per atom.
    """
    def __init__(self, keys, fill_in_field=None, extra_fields=list(),
                 partitioned=False, in_dtcm=True, prepend_n_keys=False):
        super(KeysRegion, self).__init__(
            in_dtcm=in_dtcm, unfilled=False)
        self.prepend_n_keys = prepend_n_keys
        self.partitioned = partitioned

        self.keys = keys

        if fill_in_field is None:
            self.fields = [lambda k, i: k.key()]
        else:
            self.fields = [lambda k, i: k.key(**{fill_in_field: i})]

        self.fields.extend(extra_fields)

    def _get_n_keys(self, vertex_slice):
        """Get the number of keys (not fields or values) in this slice.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms that will be represented by the region.
        """
        length = len(self.keys)
        if self.partitioned:
            length = min(length, vertex_slice.n_atoms)
        return length

    def sizeof(self, vertex_slice):
        """The size of the region in WORDS.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms that will be represented by the region.
        """
        length = self._get_n_keys(vertex_slice)
        return (length * (len(self.fields)) +
                (1 if self.prepend_n_keys else 0))

    def create_subregion(self, vertex_slice, subvertex_index):
        """Create a subregion.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms that will be represented by the region.
        subvertex_index : int
            The index of the subvertex containing this region.
        """
        # Create the data for each key
        data = []
        for k in (self.keys if not self.partitioned else
                  self.keys[vertex_slice.as_slice]):
            data.extend(f(k, subvertex_index) for f in self.fields)

        # Prepend any required additional data
        prepends = []
        if self.prepend_n_keys:
            n_keys = self._get_n_keys(vertex_slice)
            prepends.append(n_keys)

        # Create the subregion and return
        srd = np.hstack([np.array(prepends, dtype=np.uint32),
                         np.array(data, dtype=np.uint32)])
        return Subregion(srd, len(srd), False)
