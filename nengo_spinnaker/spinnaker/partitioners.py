import collections


class Slice(collections.namedtuple('Slice', 'start stop n_atoms as_slice')):
    """Represents a partition of a vertex.

    Attributes
    ----------
    start : int
        Index of the lowest represented atom in the slice.
    stop : int
        Index of the atom 1 ABOVE the highest represented atom in the slice.
    n_atoms : int
        The number of atoms represented in the slice.
    as_slice : :py:func:`slice`
        A slice object that can be used to index into arrays.
    """
    def __new__(cls, start, stop=0):
        """Create a new Slice ranging from start to stop."""
        assert 0 <= start < stop or stop == 0, "Invalid start/stop values."

        # If stop is not provided then we take it that only one atom is
        # desired.
        if stop == 0:
            stop = start + 1

        # Create the slice object representing this Slice.
        sl = slice(start, stop, 1)

        # Calculate the number of atoms
        n_atoms = stop - start

        # Create and return the slice object
        return super(cls, Slice).__new__(cls, start, stop, n_atoms, sl)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.start,
                                   self.stop)
