import mock
import pytest

from ..vertices import Vertex
from .. import regions


# Region requirements tests
def test_get_sdram_usage_for_atoms():
    """Test that the SDRAM usage for various regions is reported correctly.
    """
    rs = [
        regions.MatrixRegion(shape=(100, 5), prepends=[
            regions.MatrixRegionPrepends.SIZE,
            regions.MatrixRegionPrepends.N_ATOMS
        ]),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5))
    ]
    v = Vertex(100, '', rs)

    assert (v.get_sdram_usage_for_atoms(slice(0, 10)) ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs))


def test_get_dtcm_usage_for_regions():
    """Test that the DTCM usage for various regions is reported correctly.
    """
    rs = [
        regions.MatrixRegion(shape=(100, 5), prepends=[
            regions.MatrixRegionPrepends.SIZE,
            regions.MatrixRegionPrepends.N_ATOMS
        ]),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5))
    ]
    v = Vertex(100, '', rs)

    assert (v.get_dtcm_usage_for_atoms(slice(0, 10)) ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs))


def test_get_dtcm_usage_for_regions_non_dtcm_regions():
    """Test that the DTCM usage for various regions is reported correctly when
    some of the regions are not stored in DTCM.
    """
    rs = [
        regions.MatrixRegion(shape=(100, 5), prepends=[
            regions.MatrixRegionPrepends.SIZE,
            regions.MatrixRegionPrepends.N_ATOMS
        ]),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5), in_dtcm=False)
    ]
    v = Vertex(100, '', rs)

    assert (v.get_dtcm_usage_for_atoms(slice(0, 10)) ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs[:-1]))


def test_get_dtcm_usage_with_other_costs():
    rs = [
        regions.MatrixRegion(shape=(100, 5), prepends=[
            regions.MatrixRegionPrepends.SIZE,
            regions.MatrixRegionPrepends.N_ATOMS
        ]),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5), in_dtcm=False)
    ]
    v = Vertex(100, '', rs)
    v.get_dtcm_usage_static = lambda sl: 10*(sl.stop - sl.start)

    assert (v.get_dtcm_usage_for_atoms(slice(0, 10)) ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs[:-1]) + 4*100)


def test_get_cpu_usage_fail():
    v = Vertex(100, '', list())
    with pytest.raises(NotImplementedError):
        v.get_cpu_usage_for_atoms(slice(0, 10))


def test_get_resources_used_by_atoms():
    class SampleVertex(Vertex):
        def get_cpu_usage_for_atoms(self, vertex_slice):
            return 10

    rs = [
        regions.MatrixRegion(shape=(100, 5), prepends=[
            regions.MatrixRegionPrepends.SIZE,
            regions.MatrixRegionPrepends.N_ATOMS
        ]),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5), in_dtcm=False)
    ]

    v = SampleVertex(100, '', rs)
    resources = v.get_resources_used_by_atoms(slice(0, 10), None)

    assert (resources.dtcm.get_value() ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs[:-1]))
    assert resources.cpu.get_value() == 10


# Subvertexing
def test_get_subregions():
    """Ensure that subregions are generated correctly."""
    # Create a new Vertex with some Mock regions
    rs = [
        mock.Mock(),
        mock.Mock()
    ]
    v = Vertex(100, '', rs)

    # Create some subregions, ensure the appropriate calls are made to the
    # regions.
    v.get_subregions(0, slice(0, 10))
    for r in rs:
        r.create_subregion.assert_called_once_with(slice(0, 10), 0)

    # Assert that an error is raised if too many atoms are specified
    with pytest.raises(ValueError):
        v.get_subregions(0, slice(0, 1000))
