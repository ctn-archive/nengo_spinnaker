import math
import mock
import pytest

from nengo_spinnaker.spinnaker import partitioners
from nengo_spinnaker.spinnaker.partitioners import Slice


class TestSlices(object):
    """Check dividing slice objects."""
    def test_create(self):
        """Test creating and querying slice objects."""
        # Single atom
        sl1 = partitioners.Slice(0)
        assert sl1.start == 0
        assert sl1.stop == 1
        assert sl1.n_atoms == 1
        assert sl1.as_slice == slice(sl1.start, sl1.stop, 1)

        # Multiple atoms, offset
        sl2 = partitioners.Slice(10, 20)  # Contains 10 atoms
        assert sl2.start == 10
        assert sl2.stop == 20
        assert sl2.n_atoms == 10
        assert sl2.as_slice == slice(sl2.start, sl2.stop, 1)

        assert "10" in repr(sl2)
        assert "20" in repr(sl2)
        assert "slice" not in repr(sl2)

    def test_create_fail(self):
        with pytest.raises(ValueError):
            Slice(-1)

        with pytest.raises(ValueError):
            Slice(10, 9)

    def test_eq_hash(self):
        """Test equality and hashing for Slices."""
        sl = partitioners.Slice(0, 10)
        sls = [
            (partitioners.Slice(0, 10), True),
            (partitioners.Slice(1, 10), False),
            (partitioners.Slice(0, 9), False),
        ]

        for s, v in sls:
            assert (sl == s) is v  # Assert equivalent
            assert (hash(sl) == hash(s)) is v  # Assert hash equivs

    def test_split_into(self):
        """Check the case of dividing a slice."""
        sl1 = partitioners.Slice(0, 103)
        n_splits = 5

        # Check the start positions are sensible
        atoms_per_slice = int(math.ceil(sl1.n_atoms / float(n_splits)))
        starts = range(sl1.start, sl1.stop, atoms_per_slice)
        stops = [min((s, sl1.stop)) for s in
                 range(atoms_per_slice, sl1.stop + atoms_per_slice,
                       atoms_per_slice)]

        # Create new slices covering the range
        new_slices = sl1.split_into(n_splits)
        assert len(new_slices) == n_splits

        assert [s.start for s in new_slices] == starts
        assert [s.stop for s in new_slices] == stops
        assert all(s.n_atoms <= atoms_per_slice for s in new_slices)

    def test_split_into_offset(self):
        """Check the case of dividing a slice."""
        sl1 = partitioners.Slice(15, 103)
        n_splits = 5

        # Check the start positions are sensible
        atoms_per_slice = int(math.ceil(sl1.n_atoms / float(n_splits)))
        starts = range(sl1.start, sl1.stop, atoms_per_slice)
        stops = [min((sl1.start + s, sl1.stop)) for s in
                 range(atoms_per_slice, sl1.stop, atoms_per_slice)]

        # Create new slices covering the range
        new_slices = sl1.split_into(n_splits)
        assert len(new_slices) == n_splits

        assert [s.start for s in new_slices] == starts
        assert [s.stop for s in new_slices] == stops
        assert all(s.n_atoms <= atoms_per_slice for s in new_slices)
        assert new_slices[0].start == sl1.start
        assert new_slices[-1].stop == sl1.stop

    def test_split_into_atoms(self):
        """Edge case where there are as many splits as there are atoms."""
        s = partitioners.Slice(0, 5)  # 5 atoms
        ss = s.split_into(s.n_atoms)  # Split into 5

        assert len(ss) == s.n_atoms
        assert all([t.stop == t.start + 1 for t in ss])

    def test_split_beyond_atoms(self):
        """Check that we raise a ValueError when attempts are made to partition
        beyond the granularity of atoms.
        """
        s = partitioners.Slice(0, 5)  # 5 atoms

        with pytest.raises(ValueError):
            s.split_into(6)  # Split into 6


def test_make_partitioner_constraint():
    # Create a noddy constraint method and check that it behaves sensibly.
    constraint_measurer = mock.Mock()
    constraint_measurer.side_effect = lambda x, sl: 2*x
    constraint = partitioners.make_partitioner_constraint(
        constraint_measurer, 1000, 0.9)

    # Check the returned function for some values below, on and above the
    # limit.
    assert constraint(449, partitioners.Slice(0)) == 1  # Below the limit
    constraint_measurer.assert_called_with(449, partitioners.Slice(0))
    assert constraint(450, partitioners.Slice(1)) == 1  # On the limit
    constraint_measurer.assert_called_with(450, partitioners.Slice(1))
    assert constraint(451, partitioners.Slice(2)) == 2  # Above the limit
    assert constraint(900, partitioners.Slice(0)) == 2  # 2x limit
    assert constraint(901, partitioners.Slice(0)) == 3  # > 2x limit


class MockVertex(object):
    def __init__(self, n_atoms, dtcm_usage, cpu_usage, max_atoms=1000):
        self.n_atoms = n_atoms
        self.max_atoms = max_atoms
        self.dtcm_usage = dtcm_usage
        self.cpu_usage = cpu_usage

    def get_cpu_usage(self, vertex_slice):
        return (self.cpu_usage * vertex_slice.n_atoms /
                float(self.n_atoms))

    def get_dtcm_usage(self, vertex_slice):
        return (self.dtcm_usage * vertex_slice.n_atoms /
                float(self.n_atoms))


class TestVertexPartitioner(object):
    """Test partitioning sample vertices."""
    def test_multiple_constraints(self):
        """Test that the most heavily utilised resource is used in
        partitioning.
        """
        # Create vertices which won't fit in a single core
        vertex_cpu = MockVertex(100, 500, 1000)  # Breaks CPU requirements
        vertex_dtcm = MockVertex(100, 1000, 500)  # Breaks DTCM requirements
        vertex_atom = MockVertex(100, 1000, 500, 50)  # Breaks max_atom
        vertex_many = MockVertex(100, 1250, 2001, 101)  # Breaks on all
        vertices = [vertex_cpu, vertex_dtcm, vertex_atom, vertex_many]

        # Create constraints to test against
        cpu_constraint = partitioners.make_partitioner_constraint(
            lambda v, vs: v.get_cpu_usage(vs), 1000, 0.9)
        dtcm_constraint = partitioners.make_partitioner_constraint(
            lambda v, vs: v.get_dtcm_usage(vs), 1000, 0.9)
        atom_constraint = partitioners.make_partitioner_constraint(
            lambda v, vs: vs.n_atoms, lambda v, vs: v.max_atoms, 1.0)
        constraints = [cpu_constraint, dtcm_constraint, atom_constraint]

        # Partition and check correctness of the result
        partitions = partitioners.partition_vertices(vertices, constraints)
        assert partitions[vertex_cpu] == {partitioners.Slice(0, 50),
                                          partitioners.Slice(50, 100)}
        assert partitions[vertex_dtcm] == {partitioners.Slice(0, 50),
                                           partitioners.Slice(50, 100)}
        assert partitions[vertex_atom] == {partitioners.Slice(0, 50),
                                           partitioners.Slice(50, 100)}
        assert partitions[vertex_many] == {partitioners.Slice(0, 34),
                                           partitioners.Slice(34, 68),
                                           partitioners.Slice(68, 100)}

        # Get split vertices
        split_vertices = partitioners.get_split_vertices(partitions)
        assert (set(split_vertices[vertex_cpu]) |
                set(split_vertices[vertex_dtcm]) |
                set(split_vertices[vertex_atom]) |
                set(split_vertices[vertex_many])) == {
            partitioners.SplitVertex(vertex_cpu, partitioners.Slice(0, 50)),
            partitioners.SplitVertex(vertex_cpu, partitioners.Slice(50, 100)),
            partitioners.SplitVertex(vertex_dtcm, partitioners.Slice(0, 50)),
            partitioners.SplitVertex(vertex_dtcm, partitioners.Slice(50, 100)),
            partitioners.SplitVertex(vertex_atom, partitioners.Slice(0, 50)),
            partitioners.SplitVertex(vertex_atom, partitioners.Slice(50, 100)),
            partitioners.SplitVertex(vertex_many, partitioners.Slice(0, 34)),
            partitioners.SplitVertex(vertex_many, partitioners.Slice(34, 68)),
            partitioners.SplitVertex(vertex_many, partitioners.Slice(68, 100)),
        }

    def test_offset(self):
        """Test partitioning when one usage requirement doesn't decrease
        on a line going through the origin.
        """
        class MockVertexNL(MockVertex):
            def get_cpu_usage(self, vertex_slice):
                return super(MockVertexNL, self).get_cpu_usage(
                    vertex_slice) + 25  # Offset doesn't divide!

        # Create a vertex
        v = MockVertexNL(1000, 0, 2000)

        # Create constraints to test against
        cpu_constraint = partitioners.make_partitioner_constraint(
            lambda v, vs: v.get_cpu_usage(vs), 100, 0.9)

        # Partition
        partitions = partitioners.partition_vertices([v], [cpu_constraint])
        assert len(partitions[v]) > 2
        # TODO Come up with a better test (better partitioner also?)

    def test_fail(self):
        """Test partitioning fails when requirements can't be brought into
        desired range.
        """
        # Create a vertex
        v = MockVertex(1, 0, 100)

        # Create constraints to test against
        cpu_constraint = partitioners.make_partitioner_constraint(
            lambda v, vs: v.get_cpu_usage(vs), 100, 0.9)

        # Partition
        with pytest.raises(ValueError):
            partitioners.partition_vertices([v], [cpu_constraint])


class TestGetSplitEdges(object):
    """Check creation of "sub"edges from a list of split vertices and a list of
    edges.
    """
    def test_simple(self):
        from ..edges import Edge

        # Create a set of split vertices and a set of edges, check that we
        # generate a meaningful hypergraph.
        vs = [MockVertex(100, 0, 0), MockVertex(100, 0, 0)]
        es = [Edge(vs[0], vs[1], None)]
        split_vertices = {
            vs[0]: [partitioners.SplitVertex(vs[0], Slice(0, 50)),
                    partitioners.SplitVertex(vs[0], Slice(50, 100))],
            vs[1]: [partitioners.SplitVertex(vs[1], Slice(0, 50)),
                    partitioners.SplitVertex(vs[1], Slice(50, 100))],
        }

        # Get the split edges
        split_edges = partitioners.get_split_edges(es, split_vertices)

        assert split_edges == {
            es[0]: [
                partitioners.SplitEdge(split_vertices[vs[0]][0],
                                       split_vertices[vs[1]][0]),
                partitioners.SplitEdge(split_vertices[vs[0]][0],
                                       split_vertices[vs[1]][1]),
                partitioners.SplitEdge(split_vertices[vs[0]][1],
                                       split_vertices[vs[1]][0]),
                partitioners.SplitEdge(split_vertices[vs[0]][1],
                                       split_vertices[vs[1]][1]),
            ]
        }
