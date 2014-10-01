import collections
import mock
import numpy as np
import shutil
import tempfile

import pytest

from .. import output_generator
from .. import regions

from pacman.model.placements.placement import Placement

# Temporary definition of what the data types should look like (for the purpose
# of the methods being tested)
NengoPartitionedVertex = collections.namedtuple(
    'NengoPartitionedVertex', 'subregions timer_period')


@pytest.fixture()
def tmpdir(request):
    d = tempfile.mkdtemp()

    def rmdir():
        shutil.rmtree(d)

    request.addfinalizer(rmdir)
    return d


def test_get_region_offsets():
    # Create some simple subregions of various sizes
    srs = [
        regions.Subregion(None, 132, False),
        regions.Subregion(None, 100, True),
        regions.Subregion(None, 4, False)
    ]

    # Assert that these get the appropriate offset
    reg_offsets = output_generator.get_region_offsets(srs)

    assert reg_offsets[srs[0]] == 0
    assert reg_offsets[srs[1]] == (132) * 4
    assert reg_offsets[srs[2]] == (132 + 100) * 4


def test_get_region_offsets_with_None():
    # Create some simple subregions of various sizes
    srs = [
        regions.Subregion(None, 132, False),
        None,
        regions.Subregion(None, 4, False)
    ]

    # Assert that these get the appropriate offset
    reg_offsets = output_generator.get_region_offsets(srs)

    assert reg_offsets[srs[0]] == 0
    assert reg_offsets[srs[1]] == 132 * 4
    assert reg_offsets[srs[2]] == 132 * 4


def test_create_app_pointer_table_region():
    # Create some simple subregions of various sizes
    srs = [
        regions.Subregion(None, 132, False),
        regions.Subregion(None, 100, True),
        regions.Subregion(None, 4, False)
    ]

    magic_num = 0xAD130AD6
    version = 0x00010000
    timer_period = 1000
    app_ptr_rgn = output_generator.create_app_pointer_table_region(
        srs, magic_num=magic_num, version=version,
        timer_period=timer_period)

    # Assert that the opening words of the app_ptr_region are correct
    app_ptr_rgn_data = np.frombuffer(app_ptr_rgn.data, dtype=np.uint32)
    assert np.all(app_ptr_rgn_data[:4] ==
                  [magic_num, version, timer_period, 0x0])
    assert app_ptr_rgn_data[4] == (4 + len(srs))*4

    # Assert all the regions are included correctly
    assert np.all(
        app_ptr_rgn_data[5:] ==
        4*np.array([132, 232]) + (4 + len(srs))*4
    )


def test_create_app_pointer_table_region_with_None():
    # Create some simple subregions of various sizes
    srs = [
        regions.Subregion(None, 132, False),
        None,
        regions.Subregion(None, 4, False)
    ]

    magic_num = 0xAD130AD6
    version = 0x00010000
    timer_period = 1000
    app_ptr_rgn = output_generator.create_app_pointer_table_region(
        srs, magic_num=magic_num, version=version,
        timer_period=timer_period)

    # Assert that the opening words of the app_ptr_region are correct
    app_ptr_rgn_data = np.frombuffer(app_ptr_rgn.data, dtype=np.uint32)
    assert np.all(app_ptr_rgn_data[:4] ==
                  [magic_num, version, timer_period, 0x0])
    assert app_ptr_rgn_data[4] == (4 + len(srs))*4

    # Assert all the regions are included correctly
    assert np.all(
        app_ptr_rgn_data[5:] ==
        4*np.array([132, 132]) + (4 + len(srs))*4
    )


def test_write_core_region_files(tmpdir):
    # Create some simple subregions of various sizes
    srs = [
        regions.Subregion(np.array([1, 2, 3], dtype=np.uint32), 3, False),
        regions.Subregion(np.array([4], dtype=np.uint32), 1, False),
        regions.Subregion(np.array([5, 6], dtype=np.uint32), 2, False),
    ]

    # Ensure that these subregions are written out as files and that their base
    # address is recorded correctly.
    region_writes = output_generator.write_core_region_files(0, 1, 1, srs,
                                                             0xABCD, tmpdir)

    assert len(region_writes) == 3

    for rw in region_writes:
        assert rw.x == 0 and rw.y == 1

    # Assert sizes are correct
    assert region_writes[0].size_bytes == 3 * 4
    assert region_writes[1].size_bytes == 1 * 4
    assert region_writes[2].size_bytes == 2 * 4

    # Assert base addresses are correct
    assert region_writes[0].base_address == 0xABCD
    assert region_writes[1].base_address == 0xABCD + 3*4
    assert region_writes[2].base_address == 0xABCD + (3+1)*4

    # Read in each of the regions and assert that their data matches
    for (i, rw) in enumerate(region_writes):
        data = np.fromfile(rw.path, dtype=np.uint32)
        assert np.all(data == np.frombuffer(srs[i].data, dtype=np.uint32))


def test_write_core_region_files_unfilled(tmpdir):
    # Ensure that unfilled regions do not result in writes
    srs = [
        regions.Subregion(np.array([1, 2, 3], dtype=np.uint32), 3, False),
        regions.Subregion(np.array([4], dtype=np.uint32), 1, True),
        regions.Subregion(np.array([5, 6], dtype=np.uint32), 2, False),
    ]

    # Ensure that these subregions are written out as files and that their base
    # address is recorded correctly.
    region_writes = output_generator.write_core_region_files(0, 1, 1, srs,
                                                             0xABCD, tmpdir)

    assert len(region_writes) == 2

    for rw in region_writes:
        assert rw.x == 0 and rw.y == 1

    # Assert sizes are correct
    assert region_writes[0].size_bytes == 3 * 4
    assert region_writes[1].size_bytes == 2 * 4

    # Assert base addresses are correct
    assert region_writes[0].base_address == 0xABCD
    assert region_writes[1].base_address == 0xABCD + (3+1)*4

    # Read in each of the regions and assert that their data matches
    for (i, rw) in zip([0, 2], region_writes):
        data = np.fromfile(rw.path, dtype=np.uint32)
        assert np.all(data == np.frombuffer(srs[i].data, dtype=np.uint32))


def test_generate_data_for_placements(tmpdir):
    """Test that output files are generated for placed subvertices and that the
    base addresses used in writing the output are sensible.
    """
    # Generate placements for a single chip, ensure that the memory is
    # managed correctly.
    ds = [np.array([1, 2, 3], dtype=np.uint32),
          np.zeros(100, dtype=np.uint32),
          np.array([5, 6], dtype=np.uint32), ]
    sv = NengoPartitionedVertex(
        [regions.Subregion(d, d.size, False) for d in ds], 1000)
    placements = [Placement(sv, 0, 0, i) for i in range(17)]

    # Get the word and region writes for these placements
    r_getter_base = 0xEFEF
    r_getter_func = lambda x, y, p: r_getter_base + p
    register_getter = mock.Mock()
    register_getter.side_effect = r_getter_func
    word_writes, region_writes = output_generator.generate_data_for_placements(
        placements, 128 * 1024**2, 0, register_getter, tmpdir)

    # Ensure that these are all sensible
    # Ensure that we have the correct number of writes
    assert len(region_writes) == 17 * 4  # 17 plcmts, 4 WRITTEN regions each
    assert len(word_writes) == 17  # 17 placements

    # Build up a map of what's being written (start, start+size) and ensure no
    # overlaps
    sorted_writes = sorted(region_writes, key=lambda w: w.base_address)
    ends_starts = zip(
        [w.base_address + w.size_bytes for w in sorted_writes[:-1]],
        sorted_writes[1:]
    )
    for (end, start) in ends_starts:
        assert end <= start

    # Assert that all the data has been written correctly
    size_bytes_to_d = {4*d.size: d for d in ds}
    for rw in sorted_writes:
        try:
            d = size_bytes_to_d[rw.size_bytes]
        except KeyError:
            # Probably the system region
            continue

        dw = np.fromfile(rw.path, dtype=np.uint32).reshape(d.shape)
        assert np.all(dw == d)

    # Assert that the register getter was called with all the correct arguments
    assert register_getter.call_count == 17
    for p in range(17):
        register_getter.assert_any_call(0, 0, p)

    # Assert the returned value was retained and that an appropriate base
    # address was given.
    for ww in word_writes:
        # ***YUCK***
        i = ww.address - r_getter_base
        assert 0 <= i <= 17

        # ***YUCK***
        mem = output_generator.get_total_bytes_used(sv.subregions)
        app_ptr_table_mem = 7
        expected_value = 128*1024**2 - (i+1)*(mem + app_ptr_table_mem*4)
        assert ww.value == expected_value


def test_generate_data_for_placements_multiple_chips(tmpdir):
    # Generate placements for multiple chips, ensure that the memory is still
    # managed correctly.
    assert False, "Test not written."


def test_empty_memory_exception(tmpdir):
    """Test that an exception is thrown if there is insufficient memory."""
    sv = NengoPartitionedVertex(
        [regions.Subregion(np.zeros(100, dtype=np.uint32), 100, False)], 1000)
    p = Placement(sv, 0, 0, 1)
    register_getter = lambda x: 0xEFEF

    # Subvertex requires more memory than we have made available.
    with pytest.raises(output_generator.InsufficientMemoryError):
        output_generator.generate_data_for_placements(
            [p], 0, 0x10, register_getter, tmpdir)
