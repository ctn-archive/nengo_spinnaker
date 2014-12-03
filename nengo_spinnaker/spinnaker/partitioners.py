import collections
import math
from six import iteritems


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
        if not (0 <= start < stop) and stop != 0:
            raise ValueError("Invalid start/stop values ({} {})".format(
                start, stop))

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

    def __eq__(self, other):
        return self.start == other.start and self.stop == other.stop

    def __hash__(self):
        return hash((self.start, self.stop))

    def split_into(self, n_splits):
        """Create a new set of `n_splits` Slices which cover the same range."""
        # Fail if there are more splits than atoms
        if n_splits > self.n_atoms:
            raise ValueError(n_splits)

        # Determine how atoms there will be per split
        atoms_per_split = int(math.ceil(self.n_atoms / float(n_splits)))
        n_splits = int(math.ceil(n_splits))

        return [
            Slice(self.start + n * atoms_per_split,
                  min(self.start + (n+1) * atoms_per_split, self.stop))
            for n in range(n_splits)
        ]


class SplitVertex(collections.namedtuple('SplitVertex', 'vertex slice')):
    """Represents a partition of a vertex."""


def make_partitioner_constraint(attr_getter, limit, target_usage=1.0):
    """Create a new constraint for use in a partitioner.

    Parameters
    ----------
    attr_getter : func
        Function which accepts a vertex and a vertex slice and returns some
        indication of its usage of a given resource.
    limit : int, float or func
        Indication of how much of a resource is present on a single core.  If
        this is a function then it will be called with the vertex and a vertex
        slice to return the maximum usage of this resource.
    target_usage : float
        Indication of how much of a given resource a core should aim to use.
        For example, for CPU loadings of 90% one would pass 0.9 as this
        argument.

    Returns
    -------
    func :
        A function which can be used in a partitioner to indicate the number of
        partitions a vertex will need to be split into to if it is to respect
        the limitations of a single core.
    """
    def constraint_method(vertex, vertex_slice):
        """Report the number of partitions that will be necessary for the given
        vertex to fit within this constraint.
        """
        usage = attr_getter(vertex, vertex_slice)  # Get the current usage.
        lim = limit if not callable(limit) else limit(vertex, vertex_slice)

        if usage <= target_usage * lim:
            # No splitting is required
            return 1
        else:
            # Splitting is required.
            return int(math.ceil(usage / (target_usage * lim)))

    return constraint_method


def partition_vertices(vertices, constraints):
    """Partition the given vertices subject to the provided constraints.

    Note
    ----
    It is assumed that there are no dependencies between vertices and that
    resource usage is roughly linear with number of atoms.

    Parameters
    ----------
    vertices : list
        A list of vertices which are to be partitioned.  They must support the
        methods required by the provided constraints and have attributes
        representing the number of atoms they represent.
    constraints : list
        A list of constraints which should be used in partitioning.  Each
        constraint is a function which accepts a vertex and a slice of that
        vertex and returns an int representing the number of slices that might
        allow the vertex to be partitioned onto a series of cores.

    Returns
    -------
    dict
        A dictionary mapping from vertices to sets of slices that will fit the
        constraints for a single processing core.
    """
    # Build a mapping of vertices to slices
    partitions = dict()

    # For each vertex partition to fit within all constraints.
    for v in vertices:
        # Create an initial slice that represents the whole vertex, as a
        # starting place.
        sl = Slice(0, v.n_atoms)
        splits = [sl]

        # Partition while partitioning is required.
        i = 0
        while any(c(v, s) > 1 for c in constraints for s in splits):
            # Determine how many splits each constraint will require making
            # before the vertex will fit onto a number of cores.  Determine the
            # greatest number of splits that is required.
            n_splits = max(c(v, sl) for c in constraints) + i

            if n_splits > v.n_atoms:
                raise ValueError("Cannot partition {}".format(v))

            # Make this number of splits.
            splits = sl.split_into(n_splits)

            # Nasty, but should work for most problems
            i += 1

        # Now that partitioning is done we store the partitions that we made.
        partitions[v] = set(splits)

    return partitions


def get_split_vertices(partitions):
    """Create a mapping of vertices to of :py:class:`SplitVertex` objects from
    a mapping of vertices to slices.
    """
    split_vertices = dict()

    # For vertex create a set of split representing the slices.
    for vertex, slices in iteritems(partitions):
        split_vertices[vertex] = {SplitVertex(vertex, s) for s in slices}

    return split_vertices
