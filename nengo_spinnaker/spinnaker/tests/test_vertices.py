from ..vertices import NengoVertex
from .. import regions


## Region requirements tests
def test_get_sdram_usage_for_atoms():
    """Test that the SDRAM usage for various regions is reported correctly.
    """
    rs = [
        regions.MatrixRegion(shape=(100, 5), prepend_full_length=True,
                             prepend_n_atoms=True),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5))
    ]
    v = NengoVertex(100, '', rs)

    assert (v.get_sdram_usage_for_atoms(slice(0, 10)) ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs))


def test_get_dtcm_usage_for_regions():
    """Test that the DTCM usage for various regions is reported correctly.
    """
    rs = [
        regions.MatrixRegion(shape=(100, 5), prepend_full_length=True,
                             prepend_n_atoms=True),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5))
    ]
    v = NengoVertex(100, '', rs)

    assert (v.get_dtcm_usage_for_atoms(slice(0, 10)) ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs))


def test_get_dtcm_usage_for_regions_non_dtcm_regions():
    """Test that the DTCM usage for various regions is reported correctly when
    some of the regions are not stored in DTCM.
    """
    rs = [
        regions.MatrixRegion(shape=(100, 5), prepend_full_length=True,
                             prepend_n_atoms=True),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5), in_dtcm=False)
    ]
    v = NengoVertex(100, '', rs)

    assert (v.get_dtcm_usage_for_atoms(slice(0, 10)) ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs[:-1]))


def test_get_dtcm_usage_with_other_costs():
    rs = [
        regions.MatrixRegion(shape=(100, 5), prepend_full_length=True,
                             prepend_n_atoms=True),
        regions.MatrixRegionPartitionedByColumns(shape=(5, 100),
                                                 unfilled=True),
        regions.MatrixRegionPartitionedByRows(shape=(100, 5), in_dtcm=False)
    ]
    v = NengoVertex(100, '', rs)
    v.get_dtcm_usage_static = lambda sl: 10*(sl.stop - sl.start)

    assert (v.get_dtcm_usage_for_atoms(slice(0, 10)) ==
            4*sum(r.sizeof(slice(0, 10)) for r in rs[:-1]) + 4*100)
